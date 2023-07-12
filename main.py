import argparse
import math
import numpy as np
import os
import re
import shutil
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from models.modules import mlp
from utils import get_n_params, set_random_seeds
from utils import get_graph_from_list, get_graph_from_label
from utils import get_graph_accuracy, get_device
from utils import get_output_dir
from utils import total_correlation
from data_utils.load_dataset import prepare_dataset
from losses import calc_loss


def evaluate(model, generator, args, scaling=None):
    device = args.device
    if scaling:
        x_max, x_min, y_max, y_min = scaling

    model.eval()
    tot = [0. for _ in range(args.rollouts+1)]

    tmpa, tmpb = [], []
    for batch_data, batch_label in tqdm(generator):
        batch_graph = None
        if args.gt:
            batch_graph = batch_label[:, 0, :, -args.num_humans:]
        batch_data = batch_data.to(device)
        batch_label = batch_label[:, :, :args.num_humans, :args.feat_dim].to(device)
        preds = model.multistep_forward(batch_data[:, -args.obs_frames:, ...],
                                        batch_graph, args.rollouts)
        if scaling:
            constant = 0.3048 if args.env == 'bball' else 1.
            for i in range(args.rollouts):
                batch_label[:, i, ..., 0] = (batch_label[:, i, ..., 0] * (x_max - x_min) + x_min) * constant
                batch_label[:, i, ..., 1] = (batch_label[:, i, ..., 1] * (y_max - y_min) + y_min) * constant
                preds[i][-1][..., 0] = (preds[i][-1][..., 0] * (x_max - x_min) + x_min) * constant
                preds[i][-1][..., 1] = (preds[i][-1][..., 1] * (y_max - y_min) + y_min) * constant

        pos_losses = [(torch.sqrt(((batch_label[:, i, ..., :2] - preds[i][-1][..., :2])**2).sum(dim=-1))).sum().item()/args.num_humans for i in range(args.rollouts)]

        for j in range(args.rollouts):
            tot[j] += pos_losses[j]
            tot[-1] += pos_losses[j] / args.rollouts

    for j in range(len(tot)):
        tot[j] = tot[j] / len(generator.dataset)
    return tot

def main(args):
    args.device = device = get_device(args)
    print('using device {}'.format(args.device))
    args.output_dir = get_output_dir(args)
    print('output_dir: {}'.format(args.output_dir))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    set_random_seeds(args.randomseed)
    train_generator, val_generator, test_generator, scaling = prepare_dataset(args)
    print(len(train_generator.dataset), len(val_generator.dataset), len(test_generator.dataset))
    args.num_humans = train_generator.dataset[0][0].shape[1]
    args.feat_dim = train_generator.dataset[0][0].shape[-1]

    if args.model == 'gat':
        from models.gat import GAT
        model = GAT(args)
    elif args.model == 'rfm':
        from models.rfm import RFM
        model = RFM(args)
    elif args.model == 'imma':
        from models.imma import IMMA
        model = IMMA(args)
    else:
        assert False

    model_saved_name = '{}/best_model.pth'.format(args.output_dir)
    weights_saved_name = '{}/best_model.weights'.format(args.output_dir)

    model = model.to(device)
    print('put model to device {}'.format(device))
    n_params = get_n_params(model)
    print('# parameters: {}'.format(n_params))
    graph_acc = get_graph_accuracy(model, test_generator, args)
    print('test graph_acc before trianing:', graph_acc)
    val_losses = evaluate(model, val_generator, args, scaling)
    print('val before training:', val_losses[-1])
    test_losses = evaluate(model, test_generator, args, scaling)
    print('test_loss before training:', test_losses[-1])

    loss_fn = torch.nn.MSELoss()
    kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    best_epoch = -1
    best_val_loss = 1e9
    test_acc = -1
    dataset_size = len(train_generator.dataset)

    if args.plt:
        for skip_first in range(args.edge_types-1, -1, -1):
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            best_epoch = -1
            best_val_loss = 1e9
            test_acc = -1
            dataset_size = len(train_generator.dataset)
            # training layer "skip_first"
            model.rnn_decoder.skip_first = skip_first
            model.alpha = 0.
            for jj in range(skip_first+1, args.edge_types):
                for param in model.encoders[jj].parameters():
                    param.requires_grad = False

            for epoch in tqdm(range(1000000000)):
                model.train()
                tot_loss = 0.
                tot_kl_loss = 0.
                tot_l1_loss = 0.
                if args.plt:
                    model.alpha = min(1, (epoch+1) / 500)
                optimizer.zero_grad()
                for batch_data, batch_label in train_generator:
                    batch_graph = None
                    batch_data = batch_data.to(device)
                    batch_label = batch_label.to(device)
                    if args.gt:
                        batch_graph = batch_label[:, 0, :, -num_humans:]
                        assert batch_graph.shape[1] == num_humans

                    preds = model.multistep_forward(batch_data, batch_graph, args.rollouts)
                    losses = calc_loss(preds, batch_label, args)

                    optimizer.step()
                    tot_loss += losses.item()

                if (epoch + 1) % 10 == 0:
                    model.eval()
                    print("skip_first %d, epoch %d, loss %.3f" % (skip_first, epoch + 1, tot_loss))
                    if args.l1:
                        print('l1 loss {:.3f}'.format(tot_l1_loss))
                    if args.kl:
                        print('kl loss {:.3f}'.format(tot_kl_loss))
                    if args.plt:
                        print('alpha {:.3f}'.format(model.alpha))
                    val_losses = evaluate(model, val_generator, args, scaling)
                    test_losses = evaluate(model, test_generator, args, scaling)
                    graph_acc = get_graph_accuracy(model, test_generator, args)
                    print('test graph accuracy', np.max(graph_acc))
                    print('val_losses', val_losses[-1])

                    if args.use_wandb:
                        logs = {'epoch': epoch,
                                'graph_accuracy': graph_acc,
                                'val_loss': val_losses[-1],
                                'test_loss': test_losses[-1],
                                'train_loss': tot_loss,
                                 }
                        wandb.log(logs)

                    new_best = False
                    new_best = val_losses[-1] < best_val_loss
                    new_best_val_loss = val_losses[-1]

                    if new_best:
                        print('update best_val {} --> {}'.format(best_val_loss, new_best_val_loss))
                        best_val_loss = new_best_val_loss
                        torch.save(model, model_saved_name)
                        torch.save(model.state_dict(), weights_saved_name)
                        best_epoch = epoch
                        test_loss = evaluate(model, test_generator, args, scaling)
                    elif epoch - best_epoch > 5:
                        scheduler.step()
                        print('learning rate', scheduler.get_last_lr())

                if epoch - best_epoch >= 100 or epoch >= args.max_epoch:
                    break
    else:
        for epoch in tqdm(range(1000000000)):
            model.train()
            tot_loss = 0.
            tot_kl_loss = 0.
            tot_l1_loss = 0.
            optimizer.zero_grad()
            for batch_data, batch_label in train_generator:
                batch_graph = None
                batch_data = batch_data.to(device)
                batch_label = batch_label.to(device)
                if args.gt:
                    batch_graph = batch_label[:, 0, :, -num_humans:]
                    assert batch_graph.shape[1] == num_humans

                preds = model.multistep_forward(batch_data, batch_graph, args.rollouts)
                losses = calc_loss(preds, batch_label, args)

                optimizer.step()
                tot_loss += losses.item()

            if (epoch + 1) % 10 == 0:
                model.eval()
                print("epoch %d, loss %.3f" % (epoch + 1, tot_loss))
                if args.l1:
                    print('l1 loss {:.3f}'.format(tot_l1_loss))
                if args.kl:
                    print('kl loss {:.3f}'.format(tot_kl_loss))
                val_losses = evaluate(model, val_generator, args, scaling)
                test_losses = evaluate(model, test_generator, args, scaling)
                graph_acc = get_graph_accuracy(model, test_generator, args)
                print('test graph accuracy', np.max(graph_acc))
                print('val_losses', val_losses[-1])

                if args.use_wandb:
                    logs = {'epoch': epoch,
                            'graph_accuracy': graph_acc,
                            'val_loss': val_losses[-1],
                            'test_loss': test_losses[-1],
                            'train_loss': tot_loss,
                             }
                    wandb.log(logs)

                new_best = False
                new_best = val_losses[-1] < best_val_loss
                new_best_val_loss = val_losses[-1]

                if new_best:
                    print('update best_val {} --> {}'.format(best_val_loss, new_best_val_loss))
                    best_val_loss = new_best_val_loss
                    torch.save(model, model_saved_name)
                    torch.save(model.state_dict(), weights_saved_name)
                    best_epoch = epoch
                    test_loss = evaluate(model, test_generator, args, scaling)
                elif epoch - best_epoch > 5:
                    scheduler.step()
                    print('learning rate', scheduler.get_last_lr())

            if epoch - best_epoch >= 100 or epoch >= args.max_epoch:
                break

    mutual_info_score = -1
    if args.mi_score:
        from utils import get_mutual_info_score
        mutual_info_score = get_mutual_info_score(model, test_generator, args)
        print('mutual info score: {:.4f}'.format(mutual_info_score))

    graph_acc = get_graph_accuracy(model, test_generator, args)
    print('test graph accuracy', np.max(graph_acc))
    test_loss = evaluate(model, test_generator, args, scaling)
    print('test FDE:', test_loss[-2])
    print('test ADE:', test_loss[-1])
    test_supervised_acc = 0.
    test_loss_str = ','.join(['{:.4f}'.format(ls) for ls in test_loss])

    if args.log_file:
        with open(args.log_file, 'a+') as f:
            f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.4f},{:.4f},{}\n'.format(
                args.env,
                os.path.basename(args.dataset_path),
                args.model + '_' + args.encoder,
                'gt' if args.gt else '',
                'plt' if args.plt else '',
                'dis_obj' if args.dis_obj else '',
                'l1' if args.l1 else '',
                args.skip_first,
                'kl-{}'.format(args.kl_coef) if args.kl else '',
                mutual_info_score,
                args.lr,
                args.obs_frames,
                args.window_size,
                args.randomseed,
                args.hidden_dim,
                n_params,
                np.max(graph_acc),
                test_supervised_acc,
                test_loss_str))

    if args.visualize:
        from visualization import quick_visualization
        quick_visualization(model, test_generator, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env', type=str, choices=['socialnav', 'phase', 'bball'])
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--model', type=str, default='gat')
    parser.add_argument('--obs_frames', type=int, default=24)
    parser.add_argument('--rollouts', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--gt', default=False, action='store_true', help='use ground truth graph')
    parser.add_argument('--hidden_dim', type=int, default=24)
    parser.add_argument('--lambda1', type=float, default=0.01)
    parser.add_argument('--plt', default=False, action='store_true', help='progressive layered training')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--skip_first', default=0, type=int)
    parser.add_argument('--kl', default=False, action='store_true')
    parser.add_argument('--kl_coef', default=1e-2, type=float)
    parser.add_argument('--dis_obj', default=False, action='store_true', help='use disentanglement loss')
    parser.add_argument('--burn_in', default=False, action='store_true')
    parser.add_argument('--encoder', default='mlp', choices=['mlp', 'rnn', 'cnn'])
    parser.add_argument('--fixed', default=False, action='store_true')
    parser.add_argument('--mi_score', default=False, action='store_true')
    parser.add_argument('--randomize', default=False, action='store_true')
    parser.add_argument('--window_size', default=6, type=int)
    parser.add_argument('--edge_types', default=2, type=int)

    parser.add_argument('--project_name', type=str, default='MFCrowdSim')
    parser.add_argument('--policy', type=str, default='model_predictive_rl')
    parser.add_argument('--test_config', type=str, default=None)
    parser.add_argument('--overwrite', default=False, action='store_true')
    parser.add_argument('--l1', default=False, action='store_true')
    parser.add_argument('--weights', type=str)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--use_wandb', default=False, action='store_true')
    parser.add_argument('--test_after_every_eval', default=False, action='store_true')
    parser.add_argument('--randomseed', type=int, default=17)
    parser.add_argument('--max_epoch', type=int, default=1499)
    parser.add_argument('--log_file', type=str, default='./results.log')

    args = parser.parse_args()
    if args.use_wandb:
        run = wandb.init(project=args.project_name, reinit=True)
        run.name = '{}:{}_'.format(args.model, args.randomseed) + run.name
        print(run.name)
        wandb.config.update(args) # adds all of the arguments as config variables

    torch.set_num_threads(1)
    main(args)
