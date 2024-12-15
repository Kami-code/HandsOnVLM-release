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
git clone git@github.com:Kami-code/HandsOnVLM-release.git
cd HandsOnVLM-release
conda create -n handsonvlm python=3.10
conda activate handsonvlm
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 cuda -c pytorch -c nvidia
pip install -e .
pip install flash-attn==2.6.3 --no-build-isolation
```

## File Structure
The file structure is listed as follows:

`hoi_forecast/`: dataset structure and helper functions for handsonvlm and hoi-forecast

`handsonvlm/`: model and training code for handsonvlm

`lita/`: model and related code for lita

`llava/`: model and related code for llava

## Dataset

See [Preparing Datasets for HandsOnVLM](docs/prepare_data.md).

## Train

The HandsOnVLM model only uses one stage supervised fine-tuning. The linear projection is initialized by the LLaVA pretrained weights. The training uses 8 H100 GPUs with 80GB memory.

### Prepare public checkpoints from Vicuna, LLaVA

```Shell
git clone https://huggingface.co/lmsys/vicuna-13b-v1.3
git clone https://huggingface.co/liuhaotian/llava-pretrain-vicuna-13b-v1.3
mv vicuna-13b-v1.3 vicuna-v1-3-13b
mv llava-pretrain-vicuna-13b-v1.3 llava-vicuna-v1-3-13b-pretrain
```
Similarly for 7B checkpoints. Replace `13b` with `7b` in the above commands.

### Supervised Fine-tuning

The HandsOnVLM model can be trained using the supervised fine-tuning script [here](scripts/finetune.sh). First update LITA dataset directory in data_path (`--data_path`) and checkpoint directory (`./checkpoints`).
```Shell
sh scripts/finetune.sh
```


## Evaluation

We provide the evaluation pipeline for the EPIC-KITCHEN-100 dataset.

1. Generate LITA responses and evaluate temporal localization metrics (mIOU and P@0.5)
```Shell
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -m handsonvlm.evaluation.evaluate --model-path ./checkpoints/handsonvlm  --test-ek-version ek100
```


## Bibtex

```
@inproceedings{
}
```

## Acknowledgements

This repository employs the same code structure to that used in [LITA](https://github.com/NVlabs/LITA) and [hoi-forecast](https://github.com/stevenlsw/hoi-forecast).
