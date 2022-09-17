# Interaction Modeling with Multiplex Attention
Authors: [Fan-Yun Sun](https://cs.stanford.edu/~sunfanyun/), [Isaac Kauvar](https://ikauvar.github.io/), [Ruohan Zhang](https://ai.stanford.edu/~zharu/), [Jiachen Li](https://jiachenli94.github.io/), [Mykel Kochenderfer](https://mykel.kochenderfer.com/), [Jiajun Wu](https://jiajunwu.com/), [Nick Haber](https://ed.stanford.edu/faculty/nhaber)

Abstract: *Modeling multi-agent systems requires understanding how agents interact. Such systems are often difficult to model because they can involve a variety of types of interactions that layer together to drive rich social behavioral dynamics. Here we introduce a method for accurately modeling multi-agent systems. We present Interaction Modeling with Multiplex Attention (IMMA), a forward prediction model that uses a multiplex latent graph to represent multiple independent types of interactions and attention to account for relations of different strengths. We also introduce Progressive Layer Training, a training strategy for this architecture. We show that our approach outperforms state-of-the-art models in trajectory forecasting and relation inference, spanning three multi-agent scenarios: social navigation, cooperative task achievement, and team sports. We further demonstrate that our approach can improve zero-shot generalization and allows us to probe how different interactions impact agent behavior.*

This repository contains the codes for our paper, which is accepted at NeurIPs 2022. 
For more details, please refer to the paper ([arxiv](https://arxiv.org/abs/2208.10660), [openreview](https://openreview.net/forum?id=SeHslYhFx5-)).

## Environment Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install [socialforce](https://github.com/ChanganVR/socialforce) library
3. Install necessary packages with pip
```
pip install -r requirements.txt
```

## Data Setup

#### Social Navigation Environment ####
This is a simulated environment inspired by
[https://github.com/vita-epfl/CrowdNav](https://github.com/vita-epfl/CrowdNav).
After installing necessary dependencies, refer to the following sample commands to start the simulation.
```.bash
randomseed=17
dataset_size=100000
obs_frames=24
rollouts=10

cd data_utils/socialnav
python generate_dataset.py --dataset_size $dataset_size \
                           --randomseed $randomseed \
                           --obs_frames ${obs_frames} \
                           --rollouts ${rollouts}
```
The resulting dataset will be stored at `datasets/*.tensor`. You can make modifications to the config file `dat_utils/socialnav/configs/default.py` to change the simulation setting.

To inspect and interact with the environment (control the embodied agent with
your arrow keys):
```.bash
cd data_utils/socialnav
python human_play.py
```

#### PHASE ####
The preprocessed dataset is under `datasets/phase/collab`.
To load the dataset, refer to the function *prepare_dataset* in `data_utils/load_dataset.py`.

#### NBA dataset ####
Download the preprocessed dataset [here](https://drive.google.com/file/d/1eJbDHy3fOHfzOStf-jSuYCz_YQloQU3s/view?usp=sharing) (or run `gdown 1eJbDHy3fOHfzOStf-jSuYCz_YQloQU3s`) and place it under `datasets`. 
Alternatively, you can create your own dataset from raw sportVU logs (refer to [this repository](https://github.com/linouk23/NBA-Player-Movements) or the code under `data_utils/bball`)
To load the dataset, refer to the function *prepare_dataset* in `data_utils/load_dataset.py`.


## Training and Evaluation
Find sample commands at `run_socialnav.sh`, `run_phase.sh` and `run_bball.sh`.

## Citation
If you find the codes or paper useful for your research, please cite the following papers:
```bibtex
@article{sun2022interaction,
  title={Interaction Modeling with Multiplex Attention},
  author={Sun, Fan-Yun and Kauvar, Isaac and Zhang, Ruohan and Li, Jiachen and Kochenderfer, Mykel and Wu, Jiajun and Haber, Nick},
  journal={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## Acknwledgement 
In htis project we use (parts of) the implementations from the following works:

- [nri](https://github.com/ethanfetaya/NRI)
- [RelationalGraphLearning](https://github.com/ChanganVR/RelationalGraphLearning)

We thank the authors for open sourcing their methods.
