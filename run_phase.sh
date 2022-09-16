#!/bin/bash -ex

obs_frames=24
rollouts=10
randomseed=17
env=phase
model=gat
data_fname=datasets/socialnav_default_${randomseed}_${dataset_size}_${obs_frames}_${rollouts}.tensor
obs_frames=24
hidden_dim=128

python main.py --dataset socialnav --model $model \
               --env $env --dataset_path $data_fname \
               --hidden_dim $hidden_dim --randomseed $randomseed \
               --obs_frames $obs_frames --rollouts $rollouts --lr 1e-6 $1
