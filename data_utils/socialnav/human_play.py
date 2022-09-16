import logging
import argparse
import importlib.util
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
import logging
import random
import math
import gym
import matplotlib.lines as mlines
from matplotlib import patches
import numpy as np
from numpy.linalg import norm
import torch
from matplotlib.widgets import Slider
from matplotlib import animation
import matplotlib.pyplot as plt

from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.state import tensor_to_joint_state, JointState, ObservableState
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import point_to_segment_dist

def main(args):
    # configure logging and device
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    if args.model_dir is not None:
        if args.config is not None:
            config_file = args.config
        else:
            config_file = os.path.join(args.model_dir, 'config.py')
        if args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
            logging.info('Loaded IL weights')
        elif args.rl:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            else:
                print(os.listdir(args.model_dir))
                model_weights = os.path.join(args.model_dir, sorted(os.listdir(args.model_dir))[-1])
            logging.info('Loaded RL weights')
        else:
            model_weights = os.path.join(args.model_dir, 'best_val.pth')
            logging.info('Loaded RL weights with best VAL')

    else:
        config_file = args.config

    spec = importlib.util.spec_from_file_location('config', config_file)
    if spec is None:
        parser.error('Config file not found.')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # configure policy
    policy_config = config.PolicyConfig(args.debug)
    policy_config.obs_frames = args.obs_frames
    policy = policy_factory[policy_config.name]()
    if args.planning_depth is not None:
        policy_config.model_predictive_rl.do_action_clip = True
        policy_config.model_predictive_rl.planning_depth = args.planning_depth
    if args.planning_width is not None:
        policy_config.model_predictive_rl.do_action_clip = True
        policy_config.model_predictive_rl.planning_width = args.planning_width
    if args.sparse_search:
        policy_config.model_predictive_rl.sparse_search = True

    policy.configure(policy_config)

    # configure environment
    env_config = config.EnvConfig(args.debug)
    env_config.env.obs_frames = args.obs_frames

    if args.human_num is not None:
        env_config.sim.human_num = args.human_num
    env = gym.make('MFCrowdSim-v0')
    env.configure(env_config)
    env.center = True

    if args.square:
        env.test_scenario = 'square_crossing'
    if args.circle:
        env.test_scenario = 'circle_crossing'
    if args.test_scenario is not None:
        env.test_scenario = args.test_scenario

    robot = Robot(env_config, 'robot')
    env.set_robot(robot)
    robot.time_step = env.time_step
    robot.set_policy(policy)
    explorer = Explorer(env, robot, device, None, gamma=0.9)

    train_config = config.TrainConfig(args.debug)
    epsilon_end = train_config.train.epsilon_end
    if not isinstance(robot.policy, ORCA):
        robot.policy.set_epsilon(epsilon_end)

    policy.set_phase(args.phase)
    policy.set_device(device)
    # set safety space for ORCA in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = args.safety_space
        else:
            robot.policy.safety_space = args.safety_space
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

    policy.set_env(env)
    robot.print_info()

    rewards = []
    global_ob = env.reset(phase=args.phase, test_case=args.test_case)

    for i, friends in enumerate(env.graph):
        print('{} follows {}.'.format(i, ','.join(list(map(str, np.flatnonzero(friends).tolist())))))
    done = False
    last_pos = np.array(robot.get_position())

    action = robot.act(global_ob)
    # global_ob, _, done, info = env.step(action)
    # rewards.append(_)
    current_pos = np.array(robot.get_position())
    # while not done:
        # action = robot.act(ob)
        # ob, _, done, info = env.step(action)
        # rewards.append(_)
        # current_pos = np.array(robot.get_position())
        # logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
        # last_pos = current_pos

    # plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
    mode = 'video'
    output_file = None
    x_offset = 0.2
    y_offset = 0.4
    cmap = plt.cm.get_cmap('hsv', 10)
    robot_color = 'black'
    arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)
    display_numbers = True

    if mode == 'video':
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.tick_params(labelsize=12)
        ax.set_xlim(-11, 11)
        ax.set_ylim(-11, 11)
        ax.set_xlabel('x(m)', fontsize=14)
        ax.set_ylabel('y(m)', fontsize=14)
        show_human_start_goal = False

        # add bounary
        boundary = plt.Circle((0,0), env.circle_radius, fill=False, color='black')
        ax.add_artist(boundary)
        # add human start positions and goals
        human_colors = [cmap(i) for i in range(len(env.humans))]
        if show_human_start_goal:
            for i in range(len(env.humans)):
                human = env.humans[i]
                human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                           color=human_colors[i],
                                           marker='*', linestyle='None', markersize=8)
                ax.add_artist(human_goal)
                # human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                                            # color=human_colors[i],
                                            # marker='o', linestyle='None', markersize=8)
                # ax.add_artist(human_start)

        # add robot start position
        robot_start = mlines.Line2D([env.robot.get_start_position()[0]], [env.robot.get_start_position()[1]],
                                    color=robot_color,
                                    marker='o', linestyle='None', markersize=8)
        robot_start_position = [env.robot.get_start_position()[0], env.robot.get_start_position()[1]]
        ax.add_artist(robot_start)
        # add robot and its goal

        # robot_positions = [state[0].position for state in env.states]
        # goal = mlines.Line2D([env.robot.get_goal_position()[0]], [env.robot.get_goal_position()[1]],
                             # color=robot_color, marker='*', linestyle='None',
                             # markersize=15, label='Goal')
        robot_circle = plt.Circle((0, - env.circle_radius + robot.radius), env.robot.radius, fill=False, color=robot_color)

        # sensor_range = plt.Circle(robot_positions[0], env.robot_sensor_range, fill=False, ls='dashed')
        # ax.add_artist(sensor_range)

        ax.add_artist(robot_circle)
        # ax.add_artist(goal)
        # plt.legend([robot_circle, goal], ['Robot', 'Goal'], fontsize=14)

        # add humans and their numbers
        human_positions = [env.humans[i].get_position() for i in range(len(env.humans))]
        humans = [plt.Circle(human_positions[i], env.humans[i].radius, fill=False, color=cmap(i))
                  for i in range(len(env.humans))]

        # disable showing human numbers
        if display_numbers:
            human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] + y_offset, str(i),
                                      color='black') for i in range(len(env.humans))]

        for i, human in enumerate(humans):
            ax.add_artist(human)
            if display_numbers:
                ax.add_artist(human_numbers[i])

        # add time annotation
        time = plt.text(0.4, 0.9, 'Time: {}'.format(0), fontsize=16, transform=ax.transAxes)
        ax.add_artist(time)

        # visualize attention scores
        # if hasattr(env.robot.policy, 'get_attention_weights'):
        #     attention_scores = [
        #         plt.text(-5.5, 5 - 0.5 * i, 'Human {}: {:.2f}'.format(i + 1, env.attention_weights[0][i]),
        #                  fontsize=16) for i in range(len(env.humans))]

        # compute orientation in each step and use arrow to show the direction
        states = [(env.robot.get_full_state(),
                  [human.get_full_state() for human in env.humans],
                  [human.id for human in env.humans])]

        radius = env.robot.radius
        orientations = []
        for i in range(env.human_num + 1):
            orientation = []
            for state in states:
                agent_state = state[0] if i == 0 else state[1][i - 1]
                if env.robot.kinematics == 'unicycle' and i == 0:
                    direction = (
                    (agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(agent_state.theta),
                                                       agent_state.py + radius * np.sin(agent_state.theta)))
                else:
                    theta = np.arctan2(agent_state.vy, agent_state.vx)
                    direction = ((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                                                    agent_state.py + radius * np.sin(theta)))
                orientation.append(direction)
            orientations.append(orientation)

            if i == 0:
                arrow_color = 'black'
                arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)]
            else:
                arrows.extend(
                    [patches.FancyArrowPatch(*orientation[0], color=human_colors[i - 1], arrowstyle=arrow_style)])

        for arrow in arrows:
            ax.add_artist(arrow)

        global_step = 0

        # if len(env.trajs) != 0:
            # human_future_positions = []
            # human_future_circles = []
            # for traj in env.trajs:
                # human_future_position = [[tensor_to_joint_state(traj[step+1][0]).human_states[i].position
                                          # for step in range(env.robot.policy.planning_depth)]
                                         # for i in range(env.human_num)]
                # human_future_positions.append(human_future_position)

            # for i in range(env.human_num):
                # circles = []
                # for j in range(env.robot.policy.planning_depth):
                    # circle = plt.Circle(human_future_positions[0][i][j], env.humans[0].radius/(1.7+j), fill=False, color=cmap(i))
                    # ax.add_artist(circle)
                    # circles.append(circle)
                # human_future_circles.append(circles)

        # boundaries = []
        # for i, l in enumerate(env.centralized_planner.leaders):
            # boundary = mlines.Line2D([humans[i].center[0], humans[l].center[0]],
                                       # [humans[i].center[1], humans[l].center[1]],
                                       # linewidth=3, linestyle='-', color='red')
            # boundaries.append(boundary)
            # ax.add_artist(boundary)

        info = None
        def update(event):
            nonlocal global_step, info
            nonlocal arrows
            nonlocal global_ob, done
            # nonlocal boundaries

            if done:
                plt.close()
                return

            if event.key == 'up':
                action = ActionXY(0, 1)
                _ =  robot.act(global_ob)
            elif event.key == 'down':
                action = ActionXY(0, -1)
                _ =  robot.act(global_ob)
            elif event.key == 'right':
                action = ActionXY(1, 0)
                _ =  robot.act(global_ob)
            elif event.key == 'left':
                action = ActionXY(-1, 0)
                _ =  robot.act(global_ob)
            else:
                action = robot.act(global_ob)

            global_ob, _, done, info = env.step(action)
            rewards.append(_)
            current_pos = np.array(robot.get_position())
            logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)

            # global_step = frame_num
            # robot position
            robot_circle.center = current_pos#robot_positions[frame_num]

            # human position
            for i, human in enumerate(humans):
                human.center = env.humans[i].get_position() # human_positions[frame_num][i]
                if display_numbers:
                    human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] + y_offset))

            # for boundary in boundaries:
                # boundary.remove()
            # boundaries = []
            # for i, l in enumerate(env.centralized_planner.leaders):
                # boundary = mlines.Line2D([humans[i].center[0], humans[l].center[0]],
                                           # [humans[i].center[1], humans[l].center[1]],
                                           # linewidth=3, linestyle='-', color='red')
                # ax.add_artist(boundary)
                # boundaries.append(boundary)


            # update orientation
            for arrow in arrows:
                arrow.remove()

            orientation = []
            state = env.states[-1]
            for i in range(env.human_num + 1):
                agent_state = state[0] if i == 0 else state[1][i - 1]
                if env.robot.kinematics == 'unicycle' and i == 0:
                    direction = (
                    (agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(agent_state.theta),
                                                       agent_state.py + radius * np.sin(agent_state.theta)))
                else:
                    theta = np.arctan2(agent_state.vy, agent_state.vx)
                    direction = ((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                                                    agent_state.py + radius * np.sin(theta)))
                orientation.append(direction)

            for i in range(len(humans)+1):
                if i == 0:
                    arrows = [patches.FancyArrowPatch(*orientation[0], color='black',
                                                      arrowstyle=arrow_style)]
                else:
                    arrows.extend([patches.FancyArrowPatch(*orientation[i], color=cmap(i - 1),
                                                           arrowstyle=arrow_style)])
            for arrow in arrows:
                ax.add_artist(arrow)
                # if hasattr(env.robot.policy, 'get_attention_weights'):
                #     attention_scores[i].set_text('human {}: {:.2f}'.format(i, env.attention_weights[frame_num][i]))

            if env.centralized_planner.name == 'SocialForce':
                forces = env.centralized_planner.forces
                print(forces[0])
                print(forces[1])
                print('===========================')
            time.set_text('Time: {:.2f}'.format(global_step * env.time_step))

            # if len(env.trajs) != 0:
                # for i, circles in enumerate(human_future_circles):
                    # for j, circle in enumerate(circles):
                        # circle.center = human_future_positions[global_step][i][j]
            global_step += 1
            fig.canvas.draw()

        fig.canvas.mpl_connect('key_press_event', update)
        plt.show()

    gamma = 0.9
    cumulative_reward = sum([pow(gamma, t * robot.time_step * robot.v_pref)
         * reward for t, reward in enumerate(rewards)])
    logging.info('It takes %.2f seconds to finish. Final status is %s, cumulative_reward is %f', env.global_time, info, cumulative_reward)
    if robot.visible and info == 'reach goal':
        human_times = env.get_human_times()
        logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env', type=str, default='socialnav')
    parser.add_argument('--obs_frames', type=int, default=24)
    parser.add_argument('--config', type=str, default='configs/default.py')
    parser.add_argument('--policy', type=str, default='model_predictive_rl')
    parser.add_argument('-m', '--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--rl', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('-c', '--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--video_dir', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--human_num', type=int, default=None)
    parser.add_argument('--safety_space', type=float, default=0.2)
    parser.add_argument('--test_scenario', type=str, default=None)
    parser.add_argument('--plot_test_scenarios_hist', default=True, action='store_true')
    parser.add_argument('-d', '--planning_depth', type=int, default=None)
    parser.add_argument('-w', '--planning_width', type=int, default=None)
    parser.add_argument('--sparse_search', default=False, action='store_true')

    sys_args = parser.parse_args()

    main(sys_args)
