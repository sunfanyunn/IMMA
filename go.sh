#!/bin/bash -ex

# data_preparation

randomseed=17
dataset_size=10
obs_frames=24
rollouts=10

cd data_utils/socialnav
python generate_dataset.py --dataset_size $dataset_size \
                           --randomseed $randomseed \
                           --obs_frames ${obs_frames} \
                           --rollouts ${rollouts}
cd ../../

env=socialnav
model=imma
data_fname=datasets/socialnav_default_${randomseed}_${dataset_size}_${obs_frames}_${rollouts}.tensor
obs_frames=24
hidden_dim=96

python main.py --dataset socialnav --softmax \
               --dataset_path $data_fname \
               --hidden_dim $hidden_dim \
               --model $model --env $env --randomseed $randomseed --lr 1e-6 $1
