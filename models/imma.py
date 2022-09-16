from torch.autograd import Variable
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.functional import softmax, relu
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import *


class IMMA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.num_humans = args.num_humans
        self.obs_frames = args.obs_frames

        self.skip_first = args.skip_first
        self.edge_types = args.edge_types
        self.hidden_dim = args.hidden_dim
        self.human_state_dim = args.feat_dim
        self.input_human_state_dim = args.feat_dim
        self.tau = 0.5
        self.hard = True
        self.alpha = 1.

        self.timesteps = args.obs_frames
        self.dims = args.hidden_dim
        self.encoder_hidden = args.hidden_dim
        self.decoder_hidden = args.hidden_dim
        self.encoder_dropout = 0.
        self.decoder_dropout = 0.
        self.factor = True

        self.encoders = nn.ModuleList([MLPEncoder(self.obs_frames * self.input_human_state_dim,
                                      self.encoder_hidden,
                                      1,
                                      self.encoder_dropout, self.factor) for _ in range(self.edge_types)])

        self.rnn_decoder = RNNDecoder(n_in_node=self.decoder_hidden,
                                      edge_types=self.edge_types,
                                      n_hid=self.decoder_hidden,
                                      do_prob=self.decoder_dropout,
                                      skip_first=self.skip_first)

        self.trans = nn.Linear(self.encoder_hidden, self.decoder_hidden)
        self.out_fc3 = nn.Linear(self.decoder_hidden, self.human_state_dim)

        off_diag = np.ones([self.num_humans, self.num_humans]) - np.eye(self.num_humans)
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).to(self.device)
        self.rel_send = torch.FloatTensor(rel_send).to(self.device)

    def convert_graph(self, tmp_graph):
        batch_size = tmp_graph.shape[0]
        pred_graph = torch.zeros((batch_size, self.num_humans, self.num_humans)).to(self.device)
        for i in range(self.num_humans):
            if i > 0:
                pred_graph[:, i, 0:i] = tmp_graph[:, i, 0:i]
            pred_graph[:, i, i] = 0
            if i+1 < self.num_humans:
                pred_graph[:, i, i+1:self.num_humans] = tmp_graph[:, i, i:self.num_humans]
        return pred_graph

    def multistep_forward(self, batch_data, batch_graph, rollouts):
        # batch_size, obs_frmes, num_humans, feat_dim
        assert batch_data.requires_grad is False
        batch_size = batch_data.shape[0]
        batch_data = batch_data.permute(0, 2, 1, 3)

        pred_graphs = []
        layer_edges = []
        for layer_idx in range(self.args.edge_types):
            node_embeddings, logits = self.encoders[layer_idx](batch_data.contiguous(), self.rel_rec, self.rel_send)
            logit = logits.reshape(-1, self.num_humans, self.num_humans-1)
            edges = F.softmax(logit, dim=-1)
            pred_graphs.append(edges)
            if self.args.plt and layer_idx == self.rnn_decoder.skip_first:
                layer_edges.append(self.alpha * edges.reshape(edges.shape[0], -1, 1))
            else:
                layer_edges.append(edges.reshape(edges.shape[0], -1, 1))

        edges = torch.cat(layer_edges, dim=-1)

        if batch_graph is not None:
            # pred_graphs = []
            for layer_idx in range(edges.shape[-1]):
                # if layer_idx == 0: #edges.shape[-1]-1:
                if layer_idx == edges.shape[-1]-1:
                    idx = 0
                    for i in range(self.num_humans):
                        for j in range(self.num_humans):
                            if i == j: continue
                            if True or self.args.softmax:
                                edges[:, idx, layer_idx] = batch_graph[:, i, j]
                            idx += 1
                    pred_graphs[layer_idx] = edges[..., layer_idx].reshape(-1, self.num_humans, self.num_humans-1)

        node_embeddings = self.trans(node_embeddings)
        output = self.rnn_decoder(node_embeddings, edges,
                                  self.rel_rec, self.rel_send,
                                  pred_steps=rollouts,
                                  dynamic_graph=False,
                                  encoder=None,
                                  burn_in=self.args.burn_in,
                                  burn_in_steps=self.obs_frames)
        ret = []
        for step in range(rollouts):
            pred = self.out_fc3(output[:, :, step, :])
            if step == 0:
                pred = batch_data[:, :, -1, :self.human_state_dim] + pred
            else:
                pred = ret[-1][-1] + pred

            ret.append([pred_graphs, pred])

        return ret
