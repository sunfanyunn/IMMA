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


def get_graph_from_list(leaders):
    num_humans = len(leaders)
    normalized_A = torch.zeros((1, num_humans, num_humans))
    for follower in range(num_humans):
        leader = int(leaders[follower])
        normalized_A[0, follower, leader] += 1
    return normalized_A

def get_graph_from_label(label):
    # label (batch size, 3, num_humans, human_feat + num_humans)
    batch_size = label.shape[0]
    num_humans = label.shape[2]

    normalized_A = torch.zeros((batch_size, num_humans, num_humans))
    for i in range(batch_size):
        for follower in range(num_humans):
            leader = int(label[i, 0, follower, -1].item())
            normalized_A[i, follower, leader] += 1
    assert not normalized_A.requires_grad
    return normalized_A

def transform(obs):
    human_states_tensors = []
    for ob in obs:
        human_states_tensor = torch.Tensor([human_state.to_tuple() for human_state in ob])
        human_states_tensors.append(human_states_tensor)

    return torch.stack(human_states_tensors)

def generate_data(env,  args, obs_frames=24, num_instances=1000):
    # features are simply coordinates
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

def prepare_dataset(args):
    if args.env == 'socialnav':
        scaling = None
        all_data, all_labels = torch.load(args.dataset_path)

    elif args.env == 'phase':
        all_data = []
        all_graphs = []
        for fi in glob('../datasets/phase/collab/*.npy'):
            data = np.load(fi)
            start_idx = 0
            for cursor in range(data.shape[0]-1):
                if np.array_equal(data[cursor, :2], data[cursor+1, :2]):
                    pass
                else:
                    idata = data[start_idx:cursor+1, ...]
                    instance = np.zeros((idata.shape[0], 8, 2+3))
                    # agents
                    instance[:, 0, :2] = idata[:, 2:4]
                    instance[:, 1, :2] = idata[:, 5:7]
                    instance[:, :2, 2] = 1.
                    # balls
                    instance[:, 2, :2] = idata[:, 8:10]
                    instance[:, 3, :2] = idata[:, 10:12]
                    instance[:, 2:4, 3] = 1.
                    # landmarks
                    instance[:, 4, :2] = idata[:, 12:14]
                    instance[:, 5, :2] = idata[:, 14:16]
                    instance[:, 6, :2] = idata[:, 16:18]
                    instance[:, 7, :2] = idata[:, 18:20]
                    instance[:, 4:, 4] = 1.

                    # instance = instance[np.arange(0, instance.shape[0], 2), ...]
                    instance = instance[np.arange(0, instance.shape[0], 2), :4, :4]
                    all_data.append(instance[:50])

                    graph = np.zeros((8, 8))
                    graph[0, 2+int(idata[0, 0])] = 1.
                    graph[1, 2+int(idata[0, 0])] = 1.
                    graph[2+int(idata[0, 0]), 4+int(idata[0, 1])] = 1.
                    all_graphs.append(graph[:4, :4])

                    start_idx = cursor+1

        all_data = np.stack(all_data)
        all_graphs = np.stack(all_graphs)
        x_min, x_max = all_data[:, :, :, 0].min(), all_data[:, :, :, 0].max()
        y_min, y_max = all_data[:, :, :, 1].min(), all_data[:, :, :, 1].max()
        scaling = [x_max, x_min, y_max, y_min]
        all_data[..., 0] = (all_data[..., 0] - x_min)/(x_max - x_min)
        all_data[..., 1] = (all_data[..., 1] - y_min)/(y_max - y_min)
        all_data, all_labels = all_data[:, :24, :, :], all_data[:, 24:, :, :]
        all_data = torch.FloatTensor(all_data)
        all_graphs = np.tile(np.expand_dims(all_graphs, 1), (1, 26, 1, 1))
        all_labels = torch.FloatTensor(np.concatenate([all_labels, all_graphs], -1))
        all_labels = torch.FloatTensor(all_labels)

    elif args.env == 'bball':
        all_data = np.load('./datasets/all_data.npz.npy')
        x_min, x_max = all_data[:, :, :, 0].min(), all_data[:, :, :, 0].max()
        y_min, y_max = all_data[:, :, :, 1].min(), all_data[:, :, :, 1].max()
        scaling = [x_max, x_min, y_max, y_min]
        all_data[..., 0] = (all_data[..., 0] - x_min)/(x_max - x_min)
        all_data[..., 1] = (all_data[..., 1] - y_min)/(y_max - y_min)
        all_data[..., 5:, 3] = 1.
        all_data[..., -1, 2:] = 1.
        all_data = all_data[:args.dataset_size, ...]

        print('loaded all_data')
        all_data, all_labels = all_data[:, :24, :, :], all_data[:, 24:, :, :]
        all_data = torch.FloatTensor(all_data)
        all_labels = torch.FloatTensor(all_labels)
    else:
        assert False

    # all_data = all_data[:, -args.obs_frames:, ...]
    all_data = all_data[:, -args.obs_frames:, ...]
    # all_data = all_data.to(args.device)
    # all_labels = all_labels.to(args.device)
    # print('push entire dataset to {}'.format(args.device))
    print('data shape', all_data.shape)
    print('labels shape', all_labels.shape)

    dataset = torch.utils.data.TensorDataset(all_data, all_labels) # create your datset
    train_size = int(len(dataset) * 0.8)
    val_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size - val_size
    torch.cuda.manual_seed_all(args.randomseed)
    torch.manual_seed(args.randomseed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    if args.env == 'phase':
        print('do data augmentation ...')
        new_train_data = []
        new_train_labels = []

        from tqdm import tqdm
        for data,label in tqdm(train_dataset):
            x_min, x_max = data[0, -4:, 0].min().item(), data[0, -4:, 0].max().item()
            y_min, y_max = data[0, -4:, 1].min().item(), data[0, -4:, 1].max().item()
            origin = [(x_min+x_max)/2, (y_min+y_max/2)]
            orig_data = data.clone()
            orig_label = label.clone()

            for angle in [0, 90, 180, 270]:
                # no reflection                
                data = orig_data.clone()
                label = orig_label.clone()
                for j in range(data.shape[1]):
                    data[:, j, :2] = torch.FloatTensor(rotate(origin, data[:, j, :2].numpy(), math.radians(angle)))
                    label[:, j, :2] = torch.FloatTensor(rotate(origin, label[:, j, :2].numpy(), math.radians(angle)))
                new_train_data.append(data)
                new_train_labels.append(label)

                data = orig_data.clone()
                label = orig_label.clone()
                # reflection by x=origin[0]
                data[:, :, 0] = 2*origin[0] - data[:, :, 0]
                label[:, :, 0] = 2*origin[0] - label[:, :, 0]
                for j in range(data.shape[1]):
                    data[:, j, :2] = torch.FloatTensor(rotate(origin, data[:, j, :2].numpy(), math.radians(angle)))
                    label[:, j, :2] = torch.FloatTensor(rotate(origin, label[:, j, :2].numpy(), math.radians(angle)))
                new_train_data.append(data)
                new_train_labels.append(label)

                data = orig_data.clone()
                label = orig_label.clone()
                # reflection by y=origin[1]
                data[:, :, 1] = 2*origin[1] - data[:, :, 1]
                label[:, :, 1] = 2*origin[1] - label[:, :, 1]
                for j in range(data.shape[1]):
                    data[:, j, :2] = torch.FloatTensor(rotate(origin, data[:, j, :2].numpy(), math.radians(angle)))
                    label[:, j, :2] = torch.FloatTensor(rotate(origin, label[:, j, :2].numpy(), math.radians(angle)))
                new_train_data.append(data)
                new_train_labels.append(label)

        new_train_data = torch.stack(new_train_data)
        new_train_labels = torch.stack(new_train_labels)
        all_data = torch.FloatTensor(new_train_data)
        all_labels = torch.FloatTensor(new_train_labels)
        train_dataset = torch.utils.data.TensorDataset(all_data, all_labels) # create your datset
        print(len(train_dataset))

    # Parameters
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 1,
              'pin_memory': False,
              }
    train_generator = torch.utils.data.DataLoader(train_dataset, **params)
    val_generator = torch.utils.data.DataLoader(val_dataset, **params)
    test_generator = torch.utils.data.DataLoader(test_dataset, **params)
    return train_generator, val_generator, test_generator, scaling

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--env', type=str, default='socialnav', choices=['socialnav', 'phase', 'bball'])
    parser.add_argument('--config', type=str, default='configs/default.py')
    parser.add_argument('--dataset_size', type=int, default=100000)
    parser.add_argument('--randomseed', type=int, default=17)
    parser.add_argument('--obs_frames', type=int, default=24)
    parser.add_argument('--rollouts', type=int, default=10)
    args = parser.parse_args()
    prepare_dataset(args)
