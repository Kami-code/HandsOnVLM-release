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
git clone git@github.com:Kami-code/handsonvlm-release.git
cd handsonvlm-release
conda create -n handsonvlm python=3.10
conda activate handsonvlm
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 cuda -c pytorch -c nvidia
pip install -e .
pip install flash-attn==2.6.3 --no-build-isolation
```

2. Download the assets from
the [Google Drive](https://drive.google.com/file/d/1qc-v50eTEjpkRoWsxfqExvC1P_EKSFAa/view?usp=drive_link) and place 
the `asset` directory at the project root directory.

ln -s /ocean/projects/cis240031p/cbao/datasets/epic-kitchens-download-scripts/EPIC-KITCHENS ./data/


## File Structure
The file structure is listed as follows:

`hoi-forecast/`: dataset code and helper functions for handsonvlm and hoi-forecast

`handsonvlm/`: model and training code for handsonvlm

`lita/`: model and related code for lita

`llava/`: model and related code for llava

## Quick Start


## Bibtex

```
@inproceedings{
}
```

## Acknowledgements

This repository employs the same code structure to that used in [LITA](https://github.com/NVlabs/LITA) and [hoi-forecast](https://github.com/stevenlsw/hoi-forecast).
