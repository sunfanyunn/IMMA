import random
import os
import torch
import math
import numpy as np
from tqdm import tqdm
import collections
import numpy as np
from matplotlib.colors import LinearSegmentedColormap as lsc
import torch.nn.functional as F
import math


def get_device(args):
    if args.gpu != -1:
        device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    return device

def get_output_dir(args):
    output_dir = 'logs/' + '_'.join([os.path.basename(args.dataset_path).split('.')[0] + '_' + args.encoder,
                                     args.model,
                                     str(args.hidden_dim),
                                     str(args.lr),
                                     'burn_in' if args.burn_in else '',
                                     'kl-{}'.format(args.kl_coef) if args.kl else '',
                                     args.env,
                                     str(args.edge_types),
                                     str(args.skip_first)])
    return output_dir

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def set_random_seeds(seed):
    """
    Sets the random seeds for pytorch cpu and gpu
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None

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


def convert_graph(tmp_graph):
    batch_size = tmp_graph.shape[0]
    num_humans = tmp_graph.shape[1]
    try:
        pred_graph = torch.zeros((batch_size, num_humans, num_humans)).to(tmp_graph.get_device())
    except:
        pred_graph = torch.zeros((batch_size, num_humans, num_humans))

    for i in range(num_humans):
        if i > 0:
            pred_graph[:, i, 0:i] = tmp_graph[:, i, 0:i]
        pred_graph[:, i, i] = 0
        if i+1 < num_humans:
            pred_graph[:, i, i+1:num_humans] = tmp_graph[:, i, i:num_humans]
    return pred_graph

def get_graph_accuracy(model, generator, args):
    if args.env == 'bball' or args.model == 'naive_mlp':
        return np.nan

    model.eval()
    results = collections.defaultdict(list)
    for iidx, (batch_data, batch_label) in tqdm(enumerate(generator)):
        batch_size = batch_data.shape[0]
        batch_data = batch_data.to(args.device)
        batch_label = batch_label.to(args.device)
        batch_graph = None
        if args.gt:
            batch_graph = batch_label[:, 0, :, -model.num_humans:]

        preds = model.multistep_forward(batch_data, batch_graph, 1)
        for global_idx in range(len(preds[0][0])):
            pred_graph = preds[0][0][global_idx].to(args.device)
            if pred_graph.shape[1] != pred_graph.shape[2]:
                pred_graph = convert_graph(pred_graph)
            for i in range(pred_graph.shape[-1]):
                pred_graph[:, i, i] = 0.

            if args.env == 'phase':
                pred_graph = pred_graph[:, :2, 2:4].argmax(dim=-1) + 2
                tmp_batch_label = batch_label[:, 0, :2, -model.num_humans:].argmax(dim=-1)
                results[global_idx].append(2*(pred_graph == tmp_batch_label).sum().item())
            else:
                pred_graph = pred_graph.argmax(dim=-1)
                tmp_batch_label = batch_label[:, 0, :, -model.num_humans:].argmax(dim=-1)
                results[global_idx].append((pred_graph == tmp_batch_label).sum().item())

    rets = []
    for i in range(len(results)):
        rets.append(np.sum(results[i]) / len(generator.dataset) / model.num_humans)
    return np.max(rets)

def get_mutual_info_score(model, generator, args):
    from sklearn.metrics import normalized_mutual_info_score
    num_humans = args.num_humans
    feat_dim = args.feat_dim
    device = args.device
    results = []
    for ii in range(args.edge_types):
        for jj in range(ii+1, args.edge_types):
            tmpa, tmpb = [], []
            for batch_data, batch_label in tqdm(generator):
                batch_graph = None
                batch_data = batch_data.to(device)
                batch_label = batch_label[:, :, :num_humans, :feat_dim].to(device)
                preds = model.multistep_forward(batch_data[:, -args.obs_frames:, ...],
                                                batch_graph, args.rollouts)

                for i in range(preds[0][0][0].shape[0]):
                    _, indices = torch.sort(preds[0][0][ii][i, 0, :], dim=-1)
                    indices = indices.detach().cpu().numpy().flatten()
                    new_indices = [0 for _ in range(num_humans-1)]
                    for j in range(num_humans-1):
                        new_indices[indices[j]] = j
                    tmpa.extend(new_indices)

                    _, indices = torch.sort(preds[0][0][jj][i, 0, :], dim=-1)
                    indices = indices.detach().cpu().numpy().flatten()
                    new_indices = [0 for _ in range(num_humans-1)]
                    for j in range(num_humans-1):
                        new_indices[indices[j]] = j
                    tmpb.extend(new_indices)
            results.append(normalized_mutual_info_score(tmpa, tmpb))

    return np.mean(results)

def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))

def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)

def total_correlation(preds, dataset_size):
    eps = 1e-16
    graphs = preds[0][0]
    k = len(graphs)
    batch_size = graphs[0].shape[0]
    ##### Total correlation
    _logqz = torch.stack(graphs, dim=1)
    # print(_logqz.shape): batch_size, # layers, 5, 4
    # logqz: 2 latent variables of size 5x4
    sample = F.gumbel_softmax(_logqz, dim=-1, hard=True).unsqueeze(1)
    # sample.size: batch_size, 1, #layers, 5, 4
    _logqz = torch.log((sample * _logqz.unsqueeze(0)).sum(dim=-1) + eps).sum(dim=-1)
    # _logqz[i,j] is the log probability of sample i to be generated by input j
    # batch_size, batch_size, #layers
    logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
    # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
    logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
    # total correlation
    kl_loss = (logqz - logqz_prodmarginals).mean()
    return kl_loss
