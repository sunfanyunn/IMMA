import logging
import itertools
import torch
import torch.nn as nn
from torch.nn.functional import softmax, relu
from torch.nn import Parameter
from models.modules import mlp

class GAT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.env = args.env
        self.num_humans = args.num_humans
        self.obs_frames = args.obs_frames
        self.human_state_dim = args.feat_dim

        # architecture settings 
        self.hidden_dim = args.hidden_dim
        self.wh_dims = [4*args.hidden_dim, 2*args.hidden_dim, args.hidden_dim]
        self.w_h = mlp(self.obs_frames*self.human_state_dim, self.wh_dims, last_relu=True)

        if args.gt:
            self.final_layer = mlp(2*self.hidden_dim, [self.hidden_dim, self.hidden_dim//2, self.human_state_dim])
            self.final_layer = torch.nn.Linear(2*self.hidden_dim, self.human_state_dim)
        else:
            self.final_layer = mlp(self.hidden_dim, [self.hidden_dim, self.hidden_dim//2, self.human_state_dim])
            #torch.nn.Linear(self.hidden_dim, human_state_dim)

        self.W = Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        # for visualization

    def get_embeddings(self, human_states):
        return self.encoder(human_states)

    def encoder(self, human_states):
        # encoder
        batch_size = human_states.shape[0]
        num_humans = human_states.shape[2]
        human_states = human_states.permute(0, 2, 1, 3)
        human_states = human_states.reshape(batch_size, num_humans, -1)
        embeddings = self.w_h(human_states)
        return embeddings

    def dynamical(self, embeddings):
        # dynamical model
        X = torch.matmul(torch.matmul(embeddings, self.W), embeddings.permute(0, 2, 1))
        normalized_A = softmax(X, dim=-1)
        next_H = relu(torch.matmul(normalized_A, embeddings))
        return normalized_A, next_H

    def forward(self, human_states, batch_graph):
        # encoder
        embeddings = self.encoder(human_states)
        # dynamical
        normalized_A, next_H = self.dynamical(embeddings)

        # decoder
        if batch_graph is None:
            prev_state = human_states[:, -1, ...]
            return normalized_A, prev_state + self.final_layer(next_H)
        else:
            H = relu(torch.matmul(batch_graph, embeddings))
            prev_state = human_states[:, -1, ...]
            return normalized_A, prev_state + self.final_layer(torch.cat((H, next_H), dim=-1))

    def multistep_forward(self, batch_data, batch_graph, rollouts):

        ret = []
        for step in range(rollouts):
            tmp_graph, pred_obs = self.forward(batch_data, batch_graph)
            if step < self.args.edge_types:
                pred_graph = torch.zeros((tmp_graph.shape[0], self.num_humans, self.num_humans-1))
                for i in range(self.num_humans):
                    pred_graph[:, i, 0:i] = tmp_graph[:, i, 0:i].detach()
                    pred_graph[:, i, i:self.num_humans] = tmp_graph[:, i, i+1:self.num_humans].detach()

            ret.append([[pred_graph], pred_obs])
            batch_data = torch.cat([batch_data[:, 1:, ...], pred_obs.unsqueeze(1)], dim=1)
        return ret
