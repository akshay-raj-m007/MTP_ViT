"""Strict read-only prepared COCO adapter. SSD labels 1..4; external labels 0..3."""
import copy
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from .io import CLASSES, digest, file_hash, read
from .preprocessing import transform
from .recipes import appearance, crop_boxes, crop_window, geometry, tile_windows, validate_recipes


def category_map(categories):
    ids, names = set(), set()
    mapping = {}
    for c in categories:
        if c['id'] in ids or c['name'] in names:
            raise ValueError('Duplicate category ID/name')
        ids.add(c['id']); names.add(c['name'])
        if c['name'] == 'no_defect':
            continue
        if c['name'] not in CLASSES:
            raise ValueError(f'Unsupported foreground class: {c}')
        mapping[c['id']] = CLASSES.index(c['name'])
    if names - {'no_defect'} != set(CLASSES):
        raise ValueError('Expected exactly the four canonical defect classes')
    return mapping


def resolve_image(root, name):
    candidate = (root / name.replace('\\', '/')).resolve()
    if candidate.is_relative_to(root.resolve()) and candidate.is_file():
        return candidate
    basename = name.replace('\\', '/').rsplit('/', 1)[-1]
    matches = [p for p in (root / 'data/raw/images').rglob(basename) if p.name == basename and p.is_file()]
    if len(matches) != 1:
        raise ValueError(f'Expected one image for {name}; found {len(matches)}')
    return matches[0]


def inspect_dataset(root, prepared):
    root, prepared = Path(root).resolve(), Path(prepared).resolve()
    manifest = read(prepared / 'split_manifest.json')
    splits = {s: read(prepared / f'{s}.json') for s in ('train', 'val', 'test')}
    fingerprint = digest({'source_fingerprint': manifest['fingerprint'], 'splits': splits})
    if manifest.get('split_fingerprint') != fingerprint:
        raise ValueError('Prepared split fingerprint mismatch; no data will be regenerated here')
    manifest_ids = [im['id'] for im in manifest['images']]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError('Duplicate image IDs in manifest')
    lookup = {im['id']: im for im in manifest['images']}
    report = dict(fingerprint=fingerprint, source_fingerprint=manifest['fingerprint'],
                  splits={}, warnings=[], validation='structural only; no semantic labels inferred')
    seen_ids, seen_paths, seen_hashes, seen_groups = set(), {}, {}, {}
    all_ids = set()
    for split, data in splits.items():
        mapping = category_map(data['categories'])
        images = {im['id']: im for im in data['images']}
        if len(images) != len(data['images']) or seen_ids & images.keys():
            raise ValueError(f'Duplicate/overlapping image IDs: {split}')
        seen_ids.update(images); all_ids.update(images)
        ann_ids = set(); by_image = defaultdict(list)
        for ann in data['annotations']:
            if ann['id'] in ann_ids:
                raise ValueError(f'Duplicate annotation ID in {split}')
            ann_ids.add(ann['id'])
            if ann['image_id'] not in images:
                raise ValueError(f'Orphan annotation {ann}')
            if ann['category_id'] not in mapping:
                raise ValueError(f'Unsupported/no_defect bounding box: {ann}')
            if ann.get('iscrowd', 0) or ann.get('ignore', 0):
                raise ValueError('Crowd/ignored boxes are not supported in this experiment')
            x, y, w, h = ann['bbox']
            im = images[ann['image_id']]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in [x,y,w,h]):
                raise ValueError(f'Nonfinite bbox: {ann}')
            if w <= 0 or h <= 0 or x < 0 or y < 0 or x+w > im['width']+1e-6 or y+h > im['height']+1e-6:
                raise ValueError(f'Invalid/out-of-bounds bbox: {ann}')
            by_image[ann['image_id']].append(ann)
        for image_id, im in images.items():
            if image_id not in lookup or lookup[image_id]['split'] != split or im.get('split', split) != split:
                raise ValueError('COCO membership disagrees with split manifest')
            if not im.get('evaluation_eligible', True):
                raise ValueError('Explicitly ineligible images require a separate policy')
            path = resolve_image(root, im['file_name'])
            with Image.open(path) as opened:
                if opened.size != (im['width'], im['height']):
                    raise ValueError(f'Dimension mismatch: {path}')
                opened.load()
            actual_hash = file_hash(path)
            if lookup[image_id].get('file_sha256', actual_hash) != actual_hash:
                raise ValueError(f'Image bytes changed since preparation: {path}')
            for seen, key in [(seen_paths, str(path.resolve())), (seen_hashes, actual_hash)]:
                if key in seen and seen[key] != split:
                    raise ValueError(f'Original appears across splits: {path}')
                seen[key] = split
            for field in ('source_group', 'leakage_group', 'sha256'):
                group = lookup[image_id].get(field)
                if group is not None:
                    key = (field, str(group))
                    if key in seen_groups and seen_groups[key] != split:
                        raise ValueError(f'{field} crosses splits: {group}')
                    seen_groups[key] = split
        report['splits'][split] = dict(images=len(images), boxes=len(data['annotations']),
            empty_images=sum(not by_image[i] for i in images),
            image_classes=dict(Counter(CLASSES[c] for i in images for c in
                                       {mapping[a['category_id']] for a in by_image[i]})),
            box_classes=dict(Counter(CLASSES[mapping[a['category_id']]] for a in data['annotations'])))
    if all_ids != set(manifest_ids):
        raise ValueError('Manifest and split images differ')
    report['warnings'].append('Empty annotations are used as background targets; reviewed-clean status remains the supplied metadata.')
    report['warnings'].append('Source groups are checked when available; absent source metadata cannot establish source-independent evaluation.')
    return report


class SemDataset(Dataset):
    def __init__(self, root, prepared, split, config, training=False, indices=None, final_test=False, tiled=False):
        # Test is opened only by the explicit `experiment.py test` command, never by training or selection.
        if split not in ('train', 'val') and not (split == 'test' and final_test and not training):
            raise ValueError('This experiment exposes train and validation only; final test is deferred')
        self.root, self.config = Path(root), config
        self.coco = read(Path(prepared) / f'{split}.json')
        self.mapping = category_map(self.coco['categories'])
        self.images = self.coco['images'] if indices is None else [self.coco['images'][i] for i in indices]
        self.training = training
        self.anns = defaultdict(list)
        for a in self.coco['annotations']:
            self.anns[a['image_id']].append(a)
        # original_plus_recipes: every epoch shows each training image once unchanged and once per
        # recipe (index = view * len(images) + source). random_augmented: one random flip/jitter view.
        self.view_mode = config.get('training_view_mode', 'random_augmented')
        if self.view_mode not in ('random_augmented', 'original_plus_recipes'):
            raise ValueError(f'Unknown training view mode: {self.view_mode}')
        self.recipes = config.get('augmentation_recipes', []) if self.view_mode == 'original_plus_recipes' else []
        if self.view_mode == 'original_plus_recipes':
            validate_recipes(self.recipes)
        self.views_per_image = 1 + len(self.recipes) if training and self.recipes else 1
        # Evaluation only: each sample is (source image, window), merged per image after prediction.
        overlap = config.get('tile_overlap') if tiled and not training else None
        self.tiles = [(i, w) for i, im in enumerate(self.images)
                      for w in tile_windows(im['width'], im['height'], config['size'], overlap)] if overlap is not None else None
        self.tiled_images = {self.images[i]['id'] for i, w in self.tiles
                             if (w[2], w[3]) != (self.images[i]['width'], self.images[i]['height'])} if self.tiles else set()

    def __len__(self):
        return len(self.tiles) if self.tiles else len(self.images) * self.views_per_image

    def view_name(self, index):
        view = index // len(self.images)
        if not self.training:
            return 'original'
        if self.recipes:
            return 'original' if view == 0 else self.recipes[view - 1]['name']
        return 'random_augmented'

    def __getitem__(self, key):
        index, seed = key if isinstance(key, tuple) else (key, 0)
        if not 0 <= index < len(self):
            raise IndexError(index)
        source, view = (self.tiles[index][0], 0) if self.tiles else (index % len(self.images), index // len(self.images))
        im = self.images[source]
        anns = self.anns[im['id']]
        external = [self.mapping[a['category_id']] for a in anns]
        xywh = [a['bbox'] for a in anns]
        with Image.open(resolve_image(self.root, im['file_name'])) as image:
            image = image.convert('RGB')
        recipe = self.recipes[view - 1] if self.recipes and view > 0 else None
        window = self.tiles[index][1] if self.tiles else None
        if recipe is not None and self.config.get('native_pitch'):
            window = crop_window(image.width, image.height, self.config['size'], xywh, random.Random(seed))
        if window is not None:
            xywh, external = crop_boxes(xywh, external, window)
            image = image.crop((window[0], window[1], window[0] + window[2], window[1] + window[3]))
        if recipe is not None:
            image, xywh, _ = geometry(image, xywh, external, recipe)
            tensor, boxes, meta, canvas = transform(image, xywh.tolist(), self.config['size'], self.config['resize_mode'])
            content = (meta['left'], meta['top'], round(meta['width'] * meta['scale_x']), round(meta['height'] * meta['scale_y']))
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                canvas = appearance(canvas, recipe, content)
            tensor = torch.from_numpy(np.array(canvas, copy=True)).permute(2, 0, 1).float().div(255)
        else:
            augmentation = self.config['augmentation'] if self.training and not self.recipes else None
            tensor, boxes, meta, _ = transform(image, xywh,
                self.config['size'], self.config['resize_mode'], augmentation, seed)
        if window is not None:
            meta.update(offset_x=window[0], offset_y=window[1])
        labels = torch.tensor([c + 1 for c in external], dtype=torch.long)
        return dict(image=tensor, boxes=boxes, labels=labels, meta=meta, image_id=im['id'], view=self.view_name(index))

    def canonical_coco(self):
        data = copy.deepcopy(self.coco)
        data['categories'] = [dict(id=i, name=n) for i,n in enumerate(CLASSES)]
        allowed = {im['id'] for im in self.images}
        data['images'] = [im for im in data['images'] if im['id'] in allowed]
        data['annotations'] = [a for a in data['annotations'] if a['image_id'] in allowed]
        for a in data['annotations']:
            a['category_id'] = self.mapping[a['category_id']]
        return data


def collate(batch):
    return torch.stack([b['image'] for b in batch]), batch
