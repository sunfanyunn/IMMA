import os
import logging
import copy
import torch
from tqdm import tqdm
from crowd_sim.envs.utils.info import *


class Explorer(object):
    def __init__(self, env, robot, device, writer, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.writer = writer
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.statistics = None

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None, epoch=None,
                       print_failure=False):
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        discomfort = 0
        min_dist = []
        cumulative_rewards = []
        average_returns = []
        collision_cases = []
        timeout_cases = []

        robot_pred_pos_loss = []
        human_pred_pos_loss = []
        robot_pred_vel_loss = []
        human_pred_vel_loss = []

        if k != 1:
            pbar = tqdm(total=k)
        else:
            pbar = None

        for i in range(k):
            ob = self.env.reset(phase)

            done = False
            states = []
            actions = []
            rewards = []
            prev_traj = None
            while not done:
                action = self.robot.act(ob)

                if phase != 'train':
                    if prev_traj is not None:
                        # first dimension: before after
                        # second dimension: state, rewrad, ...
                        # third dimension: robot or human
                        pred_robot_state = prev_traj[1][0][0][:, -1, ...]
                        pred_human_state = prev_traj[1][0][1][:, -1, ...]

                        gt_robot_state = self.robot.policy.traj[0][0][0][:, -1, ...]
                        gt_human_state = self.robot.policy.traj[0][0][1][:, -1, ...]

                        robot_pred_pos_loss.append(((pred_robot_state[...,:2] - gt_robot_state[...,:2])**2).mean().item())
                        robot_pred_vel_loss.append(((pred_robot_state[...,2:4] - gt_robot_state[...,2:4])**2).mean().item())
                        human_pred_pos_loss.append(((pred_human_state[...,:2] - gt_human_state[...,:2])**2).mean().item())
                        human_pred_vel_loss.append(((pred_human_state[...,2:4] - gt_human_state[...,2:4])**2).mean().item())


                    if hasattr(self.robot.policy, 'traj'):
                        prev_traj = self.robot.policy.traj

                ob, reward, done, info = self.env.step(action)

                # last_state shall be tensor
                actions.append(action)
                states.append((self.robot.policy.last_state, self.env.graph))
                rewards.append(reward)

                if isinstance(info, Discomfort):
                    discomfort += 1
                    min_dist.append(info.min_dist)

            # print(average(robot_pred_pos_loss))
            # print(average(robot_pred_vel_loss))
            # print(average(human_pred_pos_loss))
            # print(average(human_pred_vel_loss))

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, actions, rewards, imitation_learning)

            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))
            returns = []
            for step in range(len(rewards)):
                step_return = sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                   * reward for t, reward in enumerate(rewards[step:])])
                returns.append(step_return)
            average_returns.append(average(returns))

            if pbar:
                pbar.update(1)
        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        extra_info = extra_info + '' if epoch is None else extra_info + ' in epoch {} '.format(epoch)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f},'
                     ' average return: {:.4f}'. format(phase.upper(), extra_info, success_rate, collision_rate,
                                                       avg_nav_time, average(cumulative_rewards),
                                                       average(average_returns)))
        if phase in ['val', 'test']:
            total_time = sum(success_times + collision_times + timeout_times)
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         discomfort / total_time, average(min_dist))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))
            logging.info('Robot pos prediction loss: {:.10f}'.format(average(robot_pred_pos_loss)))
            logging.info('Robot vel prediction loss: {:.10f}'.format(average(robot_pred_vel_loss)))
            logging.info('Human pos prediction loss: {:.10f}'.format(average(human_pred_pos_loss)))
            logging.info('Human vel prediction loss: {:.10f}'.format(average(human_pred_vel_loss)))

        self.statistics = success_rate, collision_rate, avg_nav_time, average(cumulative_rewards), average(average_returns)

        return self.statistics

    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states[:-1]):
            reward = rewards[i]
            state, graph = state[0], state[1]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                next_state = self.target_policy.transform(states[i+1][0])
                value = sum([pow(self.gamma, (t - i) * self.robot.time_step * self.robot.v_pref) * reward *
                             (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                next_state = states[i+1][0]
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    value = 0
            value = torch.Tensor([value]).to(self.device)
            reward = torch.Tensor([rewards[i]]).to(self.device)

            if self.target_policy.name == 'ModelPredictiveRL':
                assert len(state) == 2
                self.memory.push((state[0], state[1], graph, value, reward, next_state[0], next_state[1]))
            else:
                assert False
                self.memory.push((state, value, reward, next_state))

    def log(self, tag_prefix, global_step):
        sr, cr, time, reward, avg_return = self.statistics
        self.writer.add_scalar(tag_prefix + '/success_rate', sr, global_step)
        self.writer.add_scalar(tag_prefix + '/collision_rate', cr, global_step)
        self.writer.add_scalar(tag_prefix + '/time', time, global_step)
        self.writer.add_scalar(tag_prefix + '/reward', reward, global_step)
        self.writer.add_scalar(tag_prefix + '/avg_return', avg_return, global_step)


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
