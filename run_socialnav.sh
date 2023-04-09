#!/bin/bash -ex

randomseed=17
dataset_size=100000
obs_frames=24
rollouts=10

# data_preparation
cd data_utils/socialnav
python generate_dataset.py --dataset_size $dataset_size \
                           --randomseed $randomseed \
                           --obs_frames ${obs_frames} \
                           --rollouts ${rollouts}
cd ../../

env=socialnav
model=imma
data_fname=datasets/socialnav_default_${randomseed}_${dataset_size}_${obs_frames}_${rollouts}.tensor
hidden_dim=96

python main.py --env $env --model $model --randomseed $randomseed \
               --dataset_path $data_fname --obs_frames $obs_frames --rollouts $rollouts \ 
               --lr 1e-6 --hidden_dim $hidden_dim --plt

