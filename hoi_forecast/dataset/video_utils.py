# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/LITA/blob/main/LICENSE
import hashlib
import os

import torch
import numpy as np
from PIL import Image


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def load_image(image_file, processor, image_aspect_ratio='square') -> torch.Tensor:
    image = Image.open(image_file).convert('RGB')
    if image_aspect_ratio == 'pad':
        image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))

    # 使用共享内存来存储处理后的图像
    shm_dir = '/dev/shm/processed_images'
    os.makedirs(shm_dir, exist_ok=True)

    # 使用文件路径的哈希和文件名组合作为缓存文件名
    file_hash = hashlib.md5(image_file.encode()).hexdigest()
    base_name = os.path.basename(image_file)
    cache_file = os.path.join(shm_dir, f"{base_name}_{file_hash[:8]}.pt")

    if os.path.exists(cache_file):
        try:
            image = torch.load(cache_file, map_location='cpu', weights_only=True)
        except:
            # print(f"Failed to load {cache_file}")
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            torch.save(image, cache_file)
        return image
    else:
        processed = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        torch.save(processed, cache_file)
        return processed

def load_video_frames(visual_path, processor, image_aspect_ratio='square') -> torch.Tensor:
    assert type(visual_path) is list
    frame_paths = visual_path

    frames = []
    for frame_path in frame_paths:
        frame: torch.Tensor = load_image(frame_path, processor, image_aspect_ratio=image_aspect_ratio)
        frames.append(frame)
    return torch.stack(frames, dim=0)


def load_video(video_path, processor, num_frames, return_vid_len=False):
    import decord
    from decord import VideoReader
    decord.bridge.set_bridge("torch")
    video_reader = VideoReader(uri=video_path)

    idx = np.round(np.linspace(0, len(video_reader) - 1, num_frames)).astype(int)
    frames = video_reader.get_batch(idx)

    frames = processor.preprocess(frames, return_tensors='pt')['pixel_values']

    if return_vid_len:
        fps = video_reader.get_avg_fps()
        num_frames = len(video_reader)
        if fps > 0:
            vid_len = float(num_frames) / fps
        else:
            vid_len = 0.0
        return frames, vid_len
    else:
        return frames
