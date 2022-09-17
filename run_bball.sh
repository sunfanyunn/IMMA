#!/bin/bash -ex


env=bball
model=imma
obs_frames=24
rollouts=10
randomseed=17
hidden_dim=256
edge_types=5

CUDA_VISIBLE_DEVICES=3 python main.py --env $env --model $model --randomseed $randomseed \
               --obs_frames $obs_frames --rollouts $rollouts --edge_types $edge_types \
               --hidden_dim $hidden_dim --lr 1e-6 --plt
