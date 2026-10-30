"""Crops, tiles and prediction merging for pitch-normalised images of different sizes.

Scale carries the meaning here, so images are padded onto the canvas and never enlarged. Images larger
than the canvas train on crops around a defect and evaluate on overlapping tiles plus one whole-image
pass, which is what catches defects taller than any tile. Depends only on torch/torchvision so the SSD
experiment can load this file by path.
"""
import math
import torch
from torchvision.ops import batched_nms


def crop_window(width, height, size, boxes, rng):
    """(left, top, width, height) of a random crop holding one randomly chosen box; clean images crop anywhere."""
    box = rng.choice(list(boxes)) if len(boxes) else None
    def start(full, length, low, high):
        if full <= length: return 0
        if box is None: return rng.randint(0, full-length)
        low, high = math.floor(low), math.ceil(high)
        first, last = sorted((high-length, low))  # sorted also covers a box longer than the window
        first, last = max(0, first), min(full-length, last)
        return rng.randint(min(first, last), last)
    x, y, w, h = box if box is not None else (0, 0, 0, 0)
    crop_w, crop_h = min(width, size), min(height, size)
    return start(width, crop_w, x, x+w), start(height, crop_h, y, y+h), crop_w, crop_h


def crop_boxes(boxes, labels, window, min_visible=.5):
    """Clip xywh boxes to the window; keep only those with at least min_visible of their area inside."""
    left, top, width, height = window
    kept, kept_labels = [], []
    for (x, y, w, h), label in zip(boxes, labels):
        x0, y0 = max(x, left), max(y, top)
        x1, y1 = min(x+w, left+width), min(y+h, top+height)
        if x1 > x0 and y1 > y0 and (x1-x0)*(y1-y0) >= min_visible*w*h:
            kept.append([x0-left, y0-top, x1-x0, y1-y0]); kept_labels.append(label)
    return kept, kept_labels


def tile_windows(width, height, size, overlap):
    """Overlapping tiles covering the image, edges included, plus the whole image when it needs tiling."""
    if width <= size and height <= size: return [(0, 0, width, height)]
    if not 0 <= overlap < size: raise ValueError('Tile overlap must be smaller than the canvas')
    def starts(full):
        if full <= size: return [0]
        count = math.ceil((full-size)/(size-overlap))+1
        return [round(i*(full-size)/(count-1)) for i in range(count)]
    tiles = [(x, y, min(width, size), min(height, size)) for y in starts(height) for x in starts(width)]
    return tiles+[(0, 0, width, height)]


def merge_predictions(predictions, tiled_images, iou=.5):
    """Class-wise NMS across the tiles of each tiled image; other images pass through unchanged."""
    merged = [p for p in predictions if p['image_id'] not in tiled_images]
    groups = {}
    for p in predictions:
        if p['image_id'] in tiled_images and p['bbox'][2] > 0 and p['bbox'][3] > 0:
            groups.setdefault(p['image_id'], []).append(p)
    for items in groups.values():
        xywh = torch.tensor([p['bbox'] for p in items], dtype=torch.float32)
        boxes = torch.cat((xywh[:, :2], xywh[:, :2]+xywh[:, 2:]), 1)
        scores = torch.tensor([p['score'] for p in items], dtype=torch.float32)
        classes = torch.tensor([p['category_id'] for p in items])
        merged.extend(items[i] for i in batched_nms(boxes, scores, classes, iou).tolist())
    return merged
