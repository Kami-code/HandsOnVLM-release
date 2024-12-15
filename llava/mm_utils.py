from PIL import Image
from io import BytesIO
import base64

import torch
from transformers import StoppingCriteria

from handsonvlm.constants import HAND_TOKEN_TEMPLATE
from llava.constants import IMAGE_TOKEN_INDEX


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def process_images(images, image_processor, model_cfg):
    return image_processor(images, return_tensors='pt')['pixel_values']


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    # print("prompt = ", prompt)
    """
    ============ Tokenizer Image Token ============
    """
    """
    prompt: A chat between a curious user and an artificial intelligence assistant. 
    The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n
    Where should my hand move to if I want to {}? ASSISTANT: Sure! Here is the hand trajectory <hand_traj>.</s>
    
    """
    # print("tokenizer:", tokenizer)
    assert image_token_index == -200
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    """
    prompt.split('<image>'): ["A chat between a curious user and an artificial intelligence assistant. 
    The assistant gives helpful, detailed, and polite answers to the user's questions. USER: ", 
    '\nWhere should my hand move to if I want to {}? ASSISTANT: Sure! Here is the hand trajectory <hand_traj>.</s>']
    """

    """
    prompt_chunks: [[1, 319, 13563, 1546, 263, 12758, 1404, 322, 385, 23116, 21082, 20255, 29889, 
    450, 20255, 4076, 8444, 29892, 13173, 29892, 322, 1248, 568, 6089, 304, 278, 1404, 29915, 29879, 
    5155, 29889, 3148, 1001, 29901, 29871], 
    [1, 29871, 13, 11921, 881, 590, 1361, 4337, 304, 565, 306, 864, 304, 6571, 29973, 
    319, 1799, 9047, 13566, 29901, 18585, 29991, 2266, 338, 278, 1361, 23324, 706, 32000, 869, 2]]    
    """
    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]
    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        # If the first chunk starts with the bos token, we need to skip it
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    """
    insert_separator(X, sep) 函数的作用是在列表 X 的每两个相邻元素之间插入分隔符 sep,并返回插入分隔符后的新列表。这在需要在列表中插入特定分隔符的场景下非常有用。
    """

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    """
    input_ids: [1, 319, 13563, 1546, 263, 12758, 1404, 322, 385, 23116, 21082, 20255, 29889, 
    450, 20255, 4076, 8444, 29892, 13173, 29892, 322, 1248, 568, 6089, 304, 278, 1404, 29915, 
    29879, 5155, 29889, 3148, 1001, 29901, 29871, -200, 29871, 13, 11921, 881, 590, 1361, 4337, 
    304, 565, 306, 864, 304, 6571, 29973, 319, 1799, 9047, 13566, 29901, 29871]
    """

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def tokenizer_image_and_traj_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    hand_traj_token_idx = tokenizer(HAND_TOKEN_TEMPLATE, add_special_tokens=False).input_ids[0]
    # print("hand_traj_token_idx:", hand_traj_token_idx)

    """
    ============ Tokenizer Image Token ============
    """
    """
    prompt: A chat between a curious user and an artificial intelligence assistant. 
    The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n
    Where should my hand move to if I want to {}? ASSISTANT: Sure! Here is the hand trajectory <hand_traj>.</s>

    """
    # print("tokenizer:", tokenizer)
    assert image_token_index == -200
    # prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    split_result = prompt.split('<image>')
    # print("split_result:", split_result)
    assert len(split_result) == 2, f"Expected 2 parts in the split result, but got {len(split_result)}"

    split_result_2 = split_result[1].split('<hand_traj>')

    split_merged = [split_result[0]] + split_result_2

    # print("split_merged:", split_merged)


    """
    split_merged: ["A chat between a curious user and an artificial intelligence assistant. 
    The assistant gives helpful, detailed, and polite answers to the user's questions. USER: ", 
    '\nWhere should my hand move to if I want to {}? ASSISTANT: Sure! Here is the hand trajectory ', 
    '.</s>']
    """

    # print("split_merged:", split_merged)
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in split_merged]
    # print("prompt_chunks:", prompt_chunks)


    """
    prompt.split('<image>'): ["A chat between a curious user and an artificial intelligence assistant. 
    The assistant gives helpful, detailed, and polite answers to the user's questions. USER: ", 
    '\nWhere should my hand move to if I want to {}? ASSISTANT: Sure! Here is the hand trajectory <hand_traj>.</s>']
    """

    """
    prompt_chunks: [[1, 319, 13563, 1546, 263, 12758, 1404, 322, 385, 23116, 21082, 20255, 
    29889, 450, 20255, 4076, 8444, 29892, 13173, 29892, 322, 1248, 568, 6089, 304, 278, 1404, 
    29915, 29879, 5155, 29889, 3148, 1001, 29901, 29871], 
    [1, 29871, 13, 11921, 881, 590, 1361, 4337, 304, 565, 306, 864, 304, 6571, 29973, 319, 1799, 9047, 
    13566, 29901, 18585, 29991, 2266, 338, 278, 1361, 23324, 706, 29871], 
    [1, 869, 2]]
    """

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        # If the first chunk starts with the bos token, we need to skip it
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    """
    insert_separator(X, sep) 函数的作用是在列表 X 的每两个相邻元素之间插入分隔符 sep,并返回插入分隔符后的新列表。这在需要在列表中插入特定分隔符的场景下非常有用。
    """

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    """
    input_ids: [1, 319, 13563, 1546, 263, 12758, 1404, 322, 385, 23116, 21082, 20255, 29889, 
    450, 20255, 4076, 8444, 29892, 13173, 29892, 322, 1248, 568, 6089, 304, 278, 1404, 29915, 
    29879, 5155, 29889, 3148, 1001, 29901, 29871, -200, 29871, 13, 11921, 881, 590, 1361, 4337, 
    304, 565, 306, 864, 304, 6571, 29973, 319, 1799, 9047, 32000, 29901, 29871]
    """

    # print("input_ids:", input_ids)

    # todo: hack here to insert hand_traj_token_idx
    # todo: find the last image token index, replace it with hand_traj_token_idx

    # count the number of image token index, if has 2, replace the last one with hand_traj_token_idx

    if input_ids.count(image_token_index) > 1:
        pass_first_image = False
        for i in range(len(input_ids)):
            if input_ids[i] == image_token_index:
                if not pass_first_image:
                    pass_first_image = True
                    continue
                input_ids[i] = hand_traj_token_idx
    # print("input_ids after:", input_ids)

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]




class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0]:] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
