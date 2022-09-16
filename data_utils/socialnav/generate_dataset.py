from glob import glob
from tqdm import tqdm
import argparse
import collections
import gym
import importlib.util
import math
import math
import numpy as np
import os
import random
import sys
import torch

from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.policy.policy_factory import policy_factory

def transform(obs):
    human_states_tensors = []
    for ob in obs:
        human_states_tensor = torch.Tensor([human_state.to_tuple() for human_state in ob])
        human_states_tensors.append(human_states_tensor)

    return torch.stack(human_states_tensors)

def generate_data(env, args):
    # features are simply coordinates
    obs_frames = args.obs_frames
    num_instances = args.dataset_size
    feat_dim = 2
    all_data = []
    all_labels = []
    for test_case in tqdm(range(num_instances)):
        ob = env.reset(phase='test', test_case=test_case)
        done = False
        frame_num = np.random.randint(10, 40)
        obj_datas = []
        for i in range(100):
            action = ActionXY(0, 0)
            ob, _, _, _ = env.step(action)

            if frame_num - obs_frames < i:
                # if args.env.startswith('MY'):
                    # obj_data = []
                    # for j in range(env.num_humans):
                        # tmp = [0 for jj in range(obj_types)]
                        # tmp[env.centralized_planner.types[j]] = 1
                        # assert env.centralized_planner.types[j] < obj_types
                        # obj_data.append([env.humans[j].gx, env.humans[j].gy] + tmp)
                    # obj_datas.append(torch.FloatTensor(obj_data))

                if i == frame_num:

                    data = transform(ob[0])[..., :feat_dim]
                    all_data.append(data)

                    action = ActionXY(0, 0)
                    labels = []
                    graph_label = env.graph
                    for j in range(2*args.rollouts):
                        ob, _, _, _ = env.step(action)

                        label = transform(ob[0])[-1, ..., :feat_dim]
                        label = torch.cat([label, graph_label], dim=-1)

                        labels.append(label)
                    all_labels.append(torch.stack(labels))
                    break

    return all_data, all_labels

def get_env(args):
    spec = importlib.util.spec_from_file_location('config', args.config)
    if spec is None:
        parser.error('Config file not found.')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    train_config = config.TrainConfig(False)

    # configure policy
    policy_config = config.PolicyConfig()
    policy = policy_factory[policy_config.name]()
    if not policy.trainable:
        parser.error('Policy has to be trainable')
    policy_config.obs_frames = args.obs_frames
    policy.configure(policy_config)
    # policy.set_device(args.device)

    env_config = config.EnvConfig(False)
    env_config.env.obs_frames = args.obs_frames
    env_name = 'MFCrowdSim-v0'
    print('##### Using Env: {} #####'.format(env_name))
    env = gym.make(env_name)
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    robot.time_step = env.time_step
    env.set_robot(robot)
    robot.set_policy(policy)
    policy.set_env(env)
    env.num_humans = env_config.sim.human_num
    return env

def generate_dataset(args):
    scaling=None
    config_name = os.path.basename(args.config).split('.')[0]
    dataset_path = '../../datasets/socialnav_{}_{}_{}_{}_{}.tensor'.format(config_name,
                                                                        args.randomseed,
                                                                        args.dataset_size,
                                                                        args.obs_frames,
                                                                        args.rollouts)
    env = get_env(args)
    all_data, all_labels = generate_data(env, args)
    all_data = torch.stack(all_data)
    all_labels = torch.stack(all_labels)
    torch.save((all_data, all_labels), dataset_path)
    print('dataset saved at {}'.format(dataset_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--config', type=str, default='configs/default.py')
    parser.add_argument('--dataset_size', type=int, default=100000)
    parser.add_argument('--randomseed', type=int, default=17)
    parser.add_argument('--obs_frames', type=int, default=24)
    parser.add_argument('--rollouts', type=int, default=10)
    args = parser.parse_args()
    generate_dataset(args)
