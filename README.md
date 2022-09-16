# Interaction Modeling with Multiplex Attention 
This repository contains the codes for our paper, which is accepted at NeurIPs 2022. 
For more details, please refer to the [paper](https://arxiv.org/abs/1909.13165).

## Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install [socialforce](https://github.com/ChanganVR/socialforce) library
3. Install necessary packages with pip
```
pip install -r requirements.txt
```

## Data Setup

#### Social Navigation Environment ####
This is a simulated environment inspired by
https://github.com/vita-epfl/CrowdNav.
After installing necessary dependencies, run `run_socialnav.sh` and
the simulation would start. The resulting dataset will be stored at `datasets/*.tensor`.

To inspect and interact with the environment (control the embodied agent with
your arrow keys), do
```
cd data_utils/socialnav
python human_play.py
```
`
#### PHASE ####
The preprocessed dataset is under `datasets/phase/collab`.
Refer to `run_phase.sh`.

#### NBA dataset ####
Download the preprocessed dataset [here](https://drive.google.com/file/d/1eJbDHy3fOHfzOStf-jSuYCz_YQloQU3s/view?usp=sharing) and place it under `datasets`. 
Alternatively, you can create your own dataset from raw sportVU logs (refer to [this repository](https://github.com/linouk23/NBA-Player-Movements) or the code under `data_utils/bball`)



## Citation
If you find the codes or paper useful for your research, please cite the following papers:
```bibtex
@article{sun2022interaction,
  title={Interaction Modeling with Multiplex Attention},
  author={Sun, Fan-Yun and Kauvar, Isaac and Zhang, Ruohan and Li, Jiachen and Kochenderfer, Mykel and Wu, Jiajun and Haber, Nick},
  journal={arXiv preprint arXiv:2208.10660},
  year={2022}
}
```

## Acknwledgement 
In htis project we use (parts of) the implementations from the following works:

- [nri](https://github.com/ethanfetaya/NRI)
- [RelationalGraphLearning](https://github.com/ChanganVR/RelationalGraphLearning)

We thank the authors for open sourcing their methods.
