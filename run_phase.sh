#!/bin/bash -ex

randomseed=17
env=phase
model=imma
obs_frames=24
rollouts=10
hidden_dim=128

python main.py --model $model --env $env --randomseed $randomseed \
               --obs_frames $obs_frames --rollouts $rollouts \
               --hidden_dim $hidden_dim --lr 1e-6 --plt
