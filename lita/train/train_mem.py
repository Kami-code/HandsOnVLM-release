# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.

import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_path)
sys.path.append(os.path.join(cur_path, ".."))
sys.path.append(os.path.join(cur_path, "..", ".."))


from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()

from lita.train.train import train

if __name__ == "__main__":
    train()
