# HandsOnVLM: Vision-Language Models for Hand-Object Interaction Prediction

[[Project Page]](https://www.chenbao.tech/handsonvlm/) [[arXiv]](https://arxiv.org/abs/2412.13187) [[Paper]](https://arxiv.org/pdf/2412.13187)
-----

[HandsOnVLM: Vision-Language Models for Hand-Object Interaction Prediction](https://www.chenbao.tech/handsonvlm/), 


[Chen Bao](https://chenbao.tech), [Jiarui Xu](https://jerryxu.net/), [Xiaolong Wang](https://xiaolonw.github.io/)†, [Abhinav Gupta](https://www.cs.cmu.edu/~abhinavg/)†, [Homanga Bharadhwaj](https://homangab.github.io/)†

† Equal Advising

HandsOnVLM is a novel vision-language model for hand-object interction prediction.
This repo contains the training and inference code for HandsOnVLM.

![HandsOnVLM Teaser](docs/teaser.gif)

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

## Weights

| Model Name     |                           LLM version                           |                         Weights                         |
|----------------|:---------------------------------------------------------------:|:-------------------------------------------------------:|
| HandsOnVLM-7B  |  [Vicuna-7B-v1.3](https://huggingface.co/lmsys/vicuna-7b-v1.3)  | [Link](https://huggingface.co/Kami-code/handsonvlm-7b)  |
| HandsOnVLM-13B | [Vicuna-13B-v1.3](https://huggingface.co/lmsys/vicuna-13b-v1.3) | [Link](https://huggingface.co/Kami-code/handsonvlm-13b) |

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

The HandsOnVLM model can be trained using the supervised fine-tuning script [here](scripts/finetune.sh). To co-train with LITA's task, also update LITA dataset directory in data_path (`--data_path`) and checkpoint directory (`./checkpoints`).
```Shell
sh scripts/finetune.sh
```


## Evaluation

We provide the evaluation pipeline for the Reasoning-based EPIC-KITCHEN-100 dataset.

```Shell
python -m handsonvlm.evaluation.evaluate --model-path ./checkpoints/handsonvlm-7b
python -m handsonvlm.evaluation.evaluate --model-path ./checkpoints/handsonvlm-13b

```

## CLI Inference

Coming soon.


## Bibtex

If the contents of thie repository are helpful, please consider citing our paper 

```
@misc{bao2024handsonvlmvisionlanguagemodelshandobject,
      title={HandsOnVLM: Vision-Language Models for Hand-Object Interaction Prediction}, 
      author={Chen Bao and Jiarui Xu and Xiaolong Wang and Abhinav Gupta and Homanga Bharadhwaj},
      year={2024},
      eprint={2412.13187},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
}
```

## Acknowledgements

This repository builds upon code from [LITA](https://github.com/NVlabs/LITA) and [hoi-forecast](https://github.com/stevenlsw/hoi-forecast). We thank the authors of these papers for open-sourcing their code.
