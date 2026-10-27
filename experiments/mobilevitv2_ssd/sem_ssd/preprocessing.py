"""RGB /255, bicubic resizing, explicit geometry and local augmentation RNG."""
import random
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps


def transform(image, xywh, size, mode='resize', augmentation=None, seed=0):
    image = image.convert('RGB')
    width, height = image.size
    boxes = np.asarray(xywh, dtype=np.float32).reshape(-1, 4).copy()
    boxes[:, 2:] += boxes[:, :2]
    rng = random.Random(seed)
    if augmentation:
        if rng.random() < augmentation['hflip']:
            image = ImageOps.mirror(image)
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        if rng.random() < augmentation['vflip']:
            image = ImageOps.flip(image)
            boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
        for key, enhancer in [('brightness', ImageEnhance.Brightness), ('contrast', ImageEnhance.Contrast)]:
            amount = augmentation[key]
            image = enhancer(image).enhance(rng.uniform(1 - amount, 1 + amount))
    if mode == 'resize':
        nw = nh = size
        left = top = 0
        canvas = image.resize((size, size), Image.Resampling.BICUBIC)
    elif mode == 'letterbox':
        scale = min(size / width, size / height)
        nw, nh = max(1, round(width * scale)), max(1, round(height * scale))
        left, top = (size - nw) // 2, (size - nh) // 2
        canvas = Image.new('RGB', (size, size), (114, 114, 114))
        canvas.paste(image.resize((nw, nh), Image.Resampling.BICUBIC), (left, top))
    else:
        raise ValueError(mode)
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * (nw / width) + left
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * (nh / height) + top
    meta = dict(width=width, height=height, scale_x=nw/width, scale_y=nh/height,
                left=left, top=top, size=size, mode=mode)
    tensor = torch.from_numpy(np.array(canvas, copy=True)).permute(2, 0, 1).float().div(255)
    return tensor, torch.from_numpy(boxes).div(size), meta, canvas


def inverse(normalized_xyxy, meta, clamp=True):
    boxes = torch.as_tensor(normalized_xyxy, dtype=torch.float32).reshape(-1, 4).clone()
    boxes *= meta['size']
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - meta['left']) / meta['scale_x']
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - meta['top']) / meta['scale_y']
    if clamp:  # predictions only; training GT is never clipped
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, meta['width'])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, meta['height'])
    return boxes
