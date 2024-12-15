# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
import sys
import os

root_train_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(root_train_dir, "..", "..")
sys.path.append(root_dir)
from hoi_forecast.utils.const import HOI_PROJECT_PATH
sys.path.append(HOI_PROJECT_PATH)
from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()
from handsonvlm.train.train import train

if __name__ == "__main__":
    train()
