# HandsOnVLM: Vision-Language Models for Hand-Object Interaction Prediction

[[Project Page]](https://www.chenbao.tech/handsonvlm/) [[arXiv]](https://arxiv.org/abs/2305.05706) [[Paper]](https://www.chenbao.tech/dexart/static/paper/dexart.pdf)
-----

[HandsOnVLM: Vision-Language Models for Hand-Object Interaction Prediction](https://www.chenbao.tech/handsonvlm/), 


[Chen Bao](https://chenbao.tech), [Jiarui Xu](https://jerryxu.net/), [Xiaolong Wang](https://xiaolonw.github.io/), [Abhinav Gupta](https://www.cs.cmu.edu/~abhinavg/), [Homanga Bharadhwaj](https://homangab.github.io/)


HandsOnVLM is a novel vision-language model for hand-object interction prediction.
This repo contains the training and inference code for HandsOnVLM.

## Installation

1. Clone the repo and Create a conda env with all the Python dependencies.

```bash
git clone --recurse-submodules git@github.com:Kami-code/handsonvlm-release.git
cd handsonvlm-release
conda create -n handsonvlm python=3.10
conda activate handsonvlm
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 cuda -c pytorch -c nvidia
pip install opencv-python wandb lmdbdict tqdm scikit-learn einops pandas
```

2. Download the assets from
the [Google Drive](https://drive.google.com/file/d/1qc-v50eTEjpkRoWsxfqExvC1P_EKSFAa/view?usp=drive_link) and place 
the `asset` directory at the project root directory.

## File Structure
The file structure is listed as follows:

`hoi-forecast`: dataset code and helper functions for handsonvlm and hoi-forecast

`handsonvlm/`: model and training code for handsonvlm

`lita/`: model and related code for lita

`llava/`: model and related code for llava

## Quick Start

### Example of Evaluate on epic100


```bash
python examples/random_action.py --task_name=laptop
```

`task_name`: name of the environment [`faucet`, `laptop`, `bucket`, `toilet`]

### Example for Inference on user input

```bash
python examples/visualize_observation.py --task_name=laptop
```
`task_name`: name of the environment [`faucet`, `laptop`, `bucket`, `toilet`]


### Example for Training

```bash
/bin/bash HandsOnVLM-13B-full-task-reasoning-bs128.sh
```

## Main Results
```bash
python examples/evaluate_policy.py --task_name=laptop --checkpoint_path assets/rl_checkpoints/laptop/laptop_nopretrain_0.zip --eval_per_instance 100
python examples/evaluate_policy.py --task_name=laptop --use_test_set --checkpoint_path assets/rl_checkpoints/laptop/laptop_nopretrain_0.zip --eval_per_instance 100
```

## Visual Pretraining

## Bibtex

```
@inproceedings{
}
```

## Acknowledgements

This repository employs the same code structure to that used in [LITA](https://github.com/NVlabs/LITA) and [hoi-forecast](https://github.com/stevenlsw/hoi-forecast).
