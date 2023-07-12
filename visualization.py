from matplotlib import animation
from matplotlib import patches
from matplotlib.widgets import Slider
from numpy.linalg import norm
import argparse
import gym
import importlib.util
import logging
import math
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import seaborn as sns
import torch
from utils import *


def cmap_map(function, cmap, name='colormap_mod', N=None, gamma=None):
    """
    Modify a colormap using `function` which must operate on 3-element
    arrays of [r, g, b] values.

    You may specify the number of colors, `N`, and the opacity, `gamma`,
    value of the returned colormap. These values default to the ones in
    the input `cmap`.

    You may also specify a `name` for the colormap, so that it can be
    loaded using plt.get_cmap(name).
    """
    if N is None:
        N = cmap.N
    if gamma is None:
        gamma = cmap._gamma
    cdict = cmap._segmentdata
    # Cast the steps into lists:
    step_dict = {key: list(map(lambda x: x[0], cdict[key])) for key in cdict}
    # Now get the unique steps (first column of the arrays):
    step_list = np.unique(sum(step_dict.values(), []))
    # 'y0', 'y1' are as defined in LinearSegmentedColormap docstring:
    y0 = cmap(step_list)[:, :3]
    y1 = y0.copy()[:, :3]
    # Go back to catch the discontinuities, and place them into y0, y1
    for iclr, key in enumerate(['red', 'green', 'blue']):
        for istp, step in enumerate(step_list):
            try:
                ind = step_dict[key].index(step)
            except ValueError:
                # This step is not in this color
                continue
            y0[istp, iclr] = cdict[key][ind][1]
            y1[istp, iclr] = cdict[key][ind][2]
    # Map the colors to their new values:
    y0 = np.array(list(map(function, y0)))
    y1 = np.array(list(map(function, y1)))
    # Build the new colormap (overwriting step_dict):
    for iclr, clr in enumerate(['red', 'green', 'blue']):
        step_dict[clr] = np.vstack((step_list, y0[:, iclr], y1[:, iclr])).T
    return lsc(name, step_dict, N=N, gamma=gamma)

def plot_single_frame(batch_data, batch_label, preds, args, path=None, rollouts=10, plot_gt=True):
    batch_data = batch_data.detach().cpu()
    batch_label = batch_label.detach().cpu()
    for i in range(len(preds)):
        for j in range(len(preds[i])):
            for k in range(len(preds[i][j])):
                preds[i][j][k] = preds[i][j][k].detach().cpu()

    if args.env == 'phase':
        num_humans = 4
        args.feat_dim = feat_dim = 4
        human_radius = 0.3
        circle_radius = 1
        display_numbers = True
    elif args.env == 'bball':
        num_humans = 11
        human_radius = 0.03
        circle_radius = 1
        display_numbers = False
    else:
        num_humans = 5
        human_radius = 0.3
        circle_radius = 8
        display_numbers = True

    rewards = []
    mode = 'video'
    output_file = None
    x_offset = 0.2
    y_offset = 0.4
    linewidth = 1.7
    colors = ['r', 'g', 'b', 'purple', 'peru']
    cmap = lambda x: colors[x]
    robot_color = 'black'
    arrow_style = patches.ArrowStyle("->", head_length=3, head_width=3)

    fig, ax = plt.subplots(figsize=(20, 10))

    ax.tick_params(labelsize=12)
    human_cur_positions = [batch_data[0, -1, i, :2] for i in range(num_humans)] 
    if args.env == 'bball':
        human_colors = []
        humans = []
        for i in range(num_humans):
            if i == num_humans-1:
                assert batch_data[:, -1, i, 3] == 1
                assert batch_data[:, -1, i, 2] == 0
                human_colors.append('orange')
                humans.append(plt.Circle(human_cur_positions[i], human_radius, fill=False, color=human_colors[i]))
            else:
                team = int(batch_data[0, -1, i, 2])+5
                human_colors.append(cmap(team))
                humans.append(plt.Circle(human_cur_positions[i], human_radius, fill=False, color=human_colors[i]))
 
    else:
        human_colors = [cmap(i) for i in range(num_humans)]
        # add bounary
        boundary = plt.Circle((0,0), circle_radius, fill=False, color='black')
        ax.add_artist(boundary)

        # add humans' current position
        human_colors = [cmap(i) for i in range(num_humans)]
        humans = [plt.Circle(human_cur_positions[i], human_radius, linewidth=linewidth, fill=False, color=human_colors[i]) for i in range(num_humans)]

    # disable showing human numbers
    if display_numbers:
        human_numbers = [ax.text(humans[i].center[0] - x_offset, humans[i].center[1] + y_offset, str(i),
                                  color='black', fontsize=20) for i in range(num_humans)]

    for i, human in enumerate(humans):
        ax.add_artist(human)
        if display_numbers:
            ax.add_artist(human_numbers[i])

    for step in range(args.obs_frames-1):
        for i in range(num_humans):
            if batch_data[0, step, i, 0] == 0. and batch_data[0, step, i, 1] == 0:
                continue
            if batch_data[0, step+1, i, 0] == 0. and batch_data[0, step+1, i, 1] == 0:
                continue
            gt_traj = mlines.Line2D([batch_data[0, step, i, 0], batch_data[0, step+1, i, 0]],
                                    [batch_data[0, step, i, 1], batch_data[0, step+1, i, 1]],
                                    linestyle='dotted', linewidth=linewidth+0.1,
                                    color=human_colors[i])
            ax.add_artist(gt_traj)

    for step in range(rollouts-1):
        pred_obs = preds[step][-1]
        for i in range(num_humans):
            circle = plt.Circle(pred_obs[0, i, :2].tolist(), human_radius/(1.7+step),
                                fill=False, linewidth=linewidth, color=human_colors[i])
            ax.add_artist(circle)

            if plot_gt:
                gt_traj = mlines.Line2D([batch_label[0, step, i, 0], batch_label[0, step+1, i, 0]],
                                        [batch_label[0, step, i, 1], batch_label[0, step+1, i, 1]],
                                        linestyle='-', linewidth=linewidth,
                                        color=human_colors[i])
                ax.add_artist(gt_traj)


    textstr = ''
    fig2 = None
    if args.env == 'socialnav':
        fig2, axes = plt.subplots(1, 2, figsize=(25,10))
        label = batch_label[0, 0, :, -num_humans:]

        sns.heatmap(label.numpy(), ax=axes[0], cmap='Blues', vmin=0, vmax=1, linewidth=0.5)
        label = label.argmax(dim=-1)

        textstr += 'GROUND TRUTH:\n\n'
        for i,l in enumerate(label):
            textstr += '{} follows {}\n'.format(i, l) 

        pred_graph = preds[0][0][1]
        if pred_graph.shape[1] != pred_graph.shape[2]:
            pred_graph = convert_graph(pred_graph)
        pred_graph = pred_graph.numpy()
        pred_graph = pred_graph[0, ...]
        # pred_graph = np.ones(pred_graph.shape)
        for i in range(pred_graph.shape[0]):
            pred_graph[i, :] = pred_graph[i, :] / pred_graph[i, :].sum()
        sns.heatmap(pred_graph, ax=axes[1], cmap='Blues', vmin=0, vmax=1, linewidth=0.5, xticklabels=False, yticklabels=False)
    else:
        pass
        # print('No Ground Truth Leader Follower')

    if args.env == 'bball':
        plt.xlim(0, circle_radius)
        plt.ylim(0, circle_radius)
    else:
        ax.axis('off')
        ax.set_xlim(-circle_radius, circle_radius)
        ax.set_ylim(-circle_radius, circle_radius)
    ax.text(-25, -5, textstr, fontsize=24)
    fig.subplots_adjust(left=0.5)
    if path:
        plt.savefig(path)
    return [fig, fig2]

def plot(env, test_case, model, robot, args, path):
    def transform(obs):
        human_states_tensors = []
        for ob in obs:
            human_states_tensor = torch.Tensor([human_state.to_tuple() for human_state in ob])
            human_states_tensors.append(human_states_tensor)

        return torch.stack(human_states_tensors)

    global_ob = env.reset('test', test_case)
    num_humans = len(env.humans)
    zero_action = ActionXY(0, 0)

    rewards = []

    done = False
    last_pos = np.array(robot.get_position())

    _ = robot.act(global_ob)
    global_ob, _, done, info = env.step(zero_action)
    rewards.append(_)
    current_pos = np.array(robot.get_position())

    mode = 'video'
    output_file = None
    x_offset = 0.2
    y_offset = 0.4
    cmap = plt.cm.get_cmap('hsv', 10)
    robot_color = 'black'
    arrow_style = patches.ArrowStyle("->", head_length=2, head_width=4)
    display_numbers = True


    textstr = ''
    if args.env.startswith('MFCrowdSim'):
        for i, l in enumerate(env.centralized_planner.leaders):
            if i == l:
                textstr += '{} has fixed goal.\n'.format(i)
            else:
                textstr += '{} follows {}\n'.format(i, l) 

    elif args.env.startswith('MFMACrowdSim'):
        for i, friends in enumerate(env.centralized_planner.adjacency):
            textstr += '{} follows {}.'.format(i, ','.join(list(map(str, np.flatnonzero(friends).tolist()))))

    fig, ax = plt.subplots(figsize=(20, 10))
    plt.text(-10, 0, textstr, fontsize=24)
    plt.subplots_adjust(left=0.5)

    ax.tick_params(labelsize=12)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
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
            human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                                        color=human_colors[i],
                                        marker='o', linestyle='None', markersize=8)
            ax.add_artist(human_start)

    human_goals = []
    human_colors = [cmap(env.centralized_planner.types[i]) for i in range(len(env.humans))]
    if show_human_start_goal:
        for i in range(len(env.humans)):
            human = env.humans[i]
            human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                       color=human_colors[i],
                                       marker='*', linestyle='None', markersize=16)
            human_goals.append(human_goal)
            ax.add_artist(human_goal)

    # add robot start position
    robot_start = mlines.Line2D([env.robot.get_start_position()[0]], [env.robot.get_start_position()[1]],
                                color=robot_color,
                                marker='o', linestyle='None', markersize=8)
    robot_start_position = [env.robot.get_start_position()[0], env.robot.get_start_position()[1]]
    ax.add_artist(robot_start)
    # add robot and its goal
    robot_positions = [state[0].position for state in env.states]
    goal = mlines.Line2D([env.robot.get_goal_position()[0]], [env.robot.get_goal_position()[1]],
                         color=robot_color, marker='*', linestyle='None',
                         markersize=15, label='Goal')
    robot_circle = plt.Circle(robot_positions[0], env.robot.radius, fill=False, color=robot_color)

    ax.add_artist(robot_circle)
    ax.add_artist(goal)
    plt.legend([robot_circle, goal], ['Robot', 'Goal'], fontsize=14)

    # add humans and their numbers
    if args.env.startswith('MY'):
        human_positions = [env.humans[i].get_position() for i in range(len(env.humans))]
        humans = [plt.Circle(human_positions[i], env.humans[i].radius, fill=False, color=cmap(env.centralized_planner.types[i]))
                  for i in range(len(env.humans))]
    else:
        human_positions = [[state[1][j].position for j in range(len(env.humans))] for state in env.states]
        humans = [plt.Circle(human_positions[0][i], env.humans[i].radius, fill=False, color=cmap(i))
                  for i in range(len(env.humans))]

    # disable showing human numbers
    if display_numbers:
        human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] + y_offset, str(i),
                                  color='black', fontsize=34) for i in range(len(env.humans))]

    for i, human in enumerate(humans):
        ax.add_artist(human)
        if display_numbers:
            ax.add_artist(human_numbers[i])

    # add time annotation
    time = plt.text(0.4, 0.95, 'Time: {}'.format(0), fontsize=16, transform=ax.transAxes)
    ax.add_artist(time)

    # visualize attention scores
    # if hasattr(env.robot.policy, 'get_attention_weights'):
    #     attention_scores = [
    #         plt.text(-5.5, 5 - 0.5 * i, 'Human {}: {:.2f}'.format(i + 1, env.attention_weights[0][i]),
    #                  fontsize=16) for i in range(len(env.humans))]

    # compute orientation in each step and use arrow to show the direction
    radius = env.robot.radius
    orientations = []
    for i in range(env.human_num + 1):
        orientation = []
        for state in env.states:
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

    global_step = 0
    future_circles = []
    def update(frame_num):
        nonlocal global_step
        nonlocal info
        nonlocal global_ob, done
        nonlocal future_circles
        nonlocal human_goals
        global_step = frame_num

        global_ob, reward, done, info = env.step(zero_action)
        rewards.append(reward)
        current_pos = np.array(robot.get_position())
        logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)

        batch_data = transform(global_ob[0]).unsqueeze(0)
        assert batch_data.shape[0] == 1
        batch_graph = None
        if args.gt:
            batch_graph = env.graph.type(torch.FloatTensor) # get_graph_from_list(env.centralized_planner.leaders)

        if args.plot_pred:
            preds = model.multistep_forward(batch_data, batch_graph, args.rollouts)
            for circle in future_circles:
                circle.remove()
            future_circles = []

            for i in range(env.human_num):
                for step in range(args.rollouts):
                    pred_obs = preds[step][-1]
                    circle = plt.Circle(pred_obs[0, i, :2].tolist(), env.humans[0].radius/(1.7+step),
                                        fill=False, color=cmap(i))
                    future_circles.append(circle)
                    ax.add_artist(circle)

        # robot position
        robot_circle.center = current_pos

        # human position
        for i, human in enumerate(humans):
            human.center = env.humans[i].get_position()
            if display_numbers:
                human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] + y_offset))

        for human_goal in human_goals:
            human_goal.remove()
        human_goals = []
        for i in range(len(env.humans)):
            human = env.humans[i]
            human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                       color=human_colors[i],
                                       marker='*', linestyle='None', markersize=16)
            ax.add_artist(human_goal)
            human_goals.append(human_goal)

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
        # for arrow in arrows:
            # arrow.remove()

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
        time.set_text('Time: {:.2f}'.format(global_step * env.time_step))

    global_step += 1
    fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', update)
    time_step = 0.25
    anim = animation.FuncAnimation(fig, update, frames=100, interval=time_step * 500)
    anim.running = True

    if path is not None:
        # save as video
        ffmpeg_writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        # writer = ffmpeg_writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(path, writer=ffmpeg_writer)
        print('file saved.')
        # save output file as gif if imagemagic is installed
        # anim.save(output_file, writer='imagemagic', fps=12)
    else:
        plt.show()

    gamma = 0.9
    cumulative_reward = sum([pow(gamma, t * robot.time_step * robot.v_pref)
         * reward for t, reward in enumerate(rewards)])
    logging.info('It takes %.2f seconds to finish. Final status is %s, cumulative_reward is %f', env.global_time, info, cumulative_reward)
    if robot.visible and info == 'reach goal':
        human_times = env.get_human_times()
        logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))


def main(args):
    # configure logging and device
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)
    spec = importlib.util.spec_from_file_location('config', args.config)
    if spec is None:
        parser.error('Config file not found.')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # configure policy
    policy_config = config.PolicyConfig(args.debug)
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

    if args.human_num is not None:
        env_config.sim.human_num = args.human_num
    env = gym.make(args.env)
    env.configure(env_config)

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

    print(args.config)
    args.output_dir = ('data/gt_' if args.gt else 'data/')  + \
                       '_'.join([os.path.basename(args.config).split('.')[0],
                                          args.model + str(args.hidden_dim),
                                          'l1' if args.l1 else '',
                                          args.env,
                                          str(args.obs_frames),
                                          str(args.dataset_size),
                                          str(args.randomseed)])

    model_saved_name = '{}/best_model.pth'.format(args.output_dir)
    model = torch.load(model_saved_name)
    test_case = 402323452
    for video_id in range(20):
        test_case += video_id
        path = '{}/{}.mp4'.format(args.output_dir, str(video_id))
        plot(env, test_case, model, robot, args, path)

def visualize_trajectory(batch_data, batch_true, x_hat):
    """
    Plot the observed, ground truth, and predicted trajectories of agents.

    Args:
    batch_data: A tensor of shape (batch, timestep, agent, features) representing the observed trajectories.
    batch_true: A tensor of shape (batch, timestep, agent, features) representing the ground truth future trajectories.
    x_hat: A tensor of shape (batch, timestep, agent, features) representing the predicted future trajectories.
    """
    # Ensure numpy arrays for compatibility with matplotlib
    batch_data = batch_data.cpu().numpy()
    batch_true = batch_true.cpu().numpy()
    x_hat = x_hat.cpu().numpy()

    # The first dimension is batch size. For this visualization, we'll just look at the first example in the batch.
    batch_data = batch_data[0]
    batch_true = batch_true[0]
    x_hat = x_hat[0]

    colors = ['r', 'g', 'b', 'purple', 'peru']  # Colors for the different agents

    plt.figure(figsize=(10, 10))
    for agent in range(batch_data.shape[1]):

        # Plot observed trajectory
        plt.plot(batch_data[:, agent, 0], batch_data[:, agent, 1], '--', color=colors[agent], label='Observed' if agent == 0 else None)

        # Plot ground truth future trajectory
        plt.plot(batch_true[:, agent, 0], batch_true[:, agent, 1], '-', color=colors[agent], label='Ground truth' if agent == 0 else None)

        # Plot the initial position
        plt.scatter(batch_true[0, agent, 0], batch_true[0, agent, 1], color=colors[agent])

        # Plot predicted future trajectory
        for i in range(x_hat.shape[0]):
            plt.scatter(x_hat[i, agent, 0], x_hat[i, agent, 1], color=colors[agent], s=100*(x_hat.shape[0]-i)/x_hat.shape[0], label='Predicted' if i==0 and agent == 0 else None)

    plt.xlabel('X')
    plt.xlim([0, 1])
    plt.ylabel('Y')
    plt.ylim([0, 1])
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--dataset_size', type=int, default=100000)
    parser.add_argument('--env', type=str, default='MFCrowdSim-v0')
    parser.add_argument('--model', type=str, default='lstm_gat')
    parser.add_argument('--obs_frames', type=int, default=6)
    parser.add_argument('--rollouts', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gt', default=False, action='store_true')
    parser.add_argument('--randomseed', type=int, default=17)
    parser.add_argument('--config', type=str, default='../crowd_nav/configs/5_po_config.py')
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--l1', default=False, action='store_true')
    parser.add_argument('--plot_pred', default=False, action='store_true')

    parser.add_argument('--policy', type=str, default='model_predictive_rl')
    parser.add_argument('-m', '--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--rl', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('-v', '--visualize', default=False, action='store_true')
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
