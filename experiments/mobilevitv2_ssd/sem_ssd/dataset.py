"""Strict read-only prepared COCO adapter. SSD labels 1..4; external labels 0..3."""
import copy
import math
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from .io import CLASSES, digest, file_hash, read
from .preprocessing import transform


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
    def __init__(self, root, prepared, split, config, training=False, indices=None):
        if split not in ('train', 'val'):
            raise ValueError('This experiment exposes train and validation only; final test is deferred')
        self.root, self.config = Path(root), config
        self.coco = read(Path(prepared) / f'{split}.json')
        self.mapping = category_map(self.coco['categories'])
        self.images = self.coco['images'] if indices is None else [self.coco['images'][i] for i in indices]
        self.training = training
        self.anns = defaultdict(list)
        for a in self.coco['annotations']:
            self.anns[a['image_id']].append(a)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, key):
        index, seed = key if isinstance(key, tuple) else (key, 0)
        im = self.images[index]
        anns = self.anns[im['id']]
        with Image.open(resolve_image(self.root, im['file_name'])) as image:
            tensor, boxes, meta, _ = transform(image, [a['bbox'] for a in anns],
                self.config['size'], self.config['resize_mode'],
                self.config['augmentation'] if self.training else None, seed)
        labels = torch.tensor([self.mapping[a['category_id']] + 1 for a in anns], dtype=torch.long)
        return dict(image=tensor, boxes=boxes, labels=labels, meta=meta, image_id=im['id'])

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
