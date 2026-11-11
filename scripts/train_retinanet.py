"""Two-stage RetinaNet (ResNet-50 + FPN) training, a CNN baseline beside YOLOS and SSD-MobileViTv2.

Reuses the YOLOS data pipeline unchanged (recipes, pitch-normalised crops, tiled evaluation, weighted
sampling), so only the detector differs. Stage A trains the FPN and heads with the ResNet frozen; stage B
also unfreezes the last ResNet block. With fixed_schedule set, every epoch's validation predictions are
saved for cv5.py to pool across folds. Safe to rerun with --resume.
"""

# Allow direct execution without installing the project package.
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import math
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.transforms.v2 import functional as F
from transformers import get_cosine_schedule_with_warmup
from sem_yolos.common import ROOT, CLASSES, read, write, prepared_dir, normalized_config, prepared_fingerprint
from sem_yolos.data import CocoDetectionDataset, collate_fn, sample_weights, inverse_boxes
from sem_yolos.metrics import coco_ap
from sem_yolos.windows import merge_predictions


class TensorTargets:
    """Stands in for the YOLOS image processor: a 0-1 image tensor and torchvision xyxy targets.

    RetinaNet normalises the image itself, and its transform is fixed to the canvas size, so nothing is resized here.
    """
    def __call__(self, images, annotations, do_resize=False, return_tensors='pt'):
        pixels = F.to_dtype(F.to_image(images), torch.float32, scale=True)
        xywh = torch.tensor([a['bbox'] for a in annotations['annotations']], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor([a['category_id'] for a in annotations['annotations']], dtype=torch.int64)
        keep = (xywh[:, 2] > 1e-2) & (xywh[:, 3] > 1e-2)  # torchvision rejects zero-size boxes
        boxes = torch.cat((xywh[keep, :2], xywh[keep, :2]+xywh[keep, 2:]), 1)
        return dict(pixel_values=pixels[None], labels=[dict(boxes=boxes, labels=labels[keep])])


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(cfg):
    # Always built from the COCO weights, also on resume: torchvision only uses frozen BatchNorm for pretrained models.
    model = retinanet_resnet50_fpn(weights=getattr(RetinaNet_ResNet50_FPN_Weights, cfg['weights']), weights_backbone=None)
    head = model.head.classification_head
    # New four-class classifier; box regression, FPN and backbone keep their COCO weights.
    model.head.classification_head = RetinaNetClassificationHead(model.backbone.out_channels, head.num_anchors, len(CLASSES))
    # One fixed canvas size, so pitch-normalised images are never rescaled inside the model.
    model.transform = GeneralizedRCNNTransform(cfg['size'], cfg['size'], [.485, .456, .406], [.229, .224, .225])
    model.score_thresh, model.nms_thresh = cfg['score_floor'], cfg['nms_iou']
    model.topk_candidates, model.detections_per_img = cfg['pre_nms_topk'], cfg['max_detections']
    return model


def configure_stage(model, stage, prefix):
    for name, p in model.named_parameters():
        p.requires_grad = not name.startswith('backbone.body.') or (stage == 'B' and name.startswith(prefix))
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    report = dict(stage=stage, names=names, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  total_parameters=sum(p.numel() for p in model.parameters()))
    print({k: v for k, v in report.items() if k != 'names'})
    return report


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval(); predictions = []
    for batch in loader:
        outputs = model(list(batch['pixel_values'].to(device)))
        for output, meta, image_id in zip(outputs, batch['meta'], batch['image_ids']):
            boxes = inverse_boxes(output['boxes'].cpu(), meta)
            for (x, y, x2, y2), score, label in zip(boxes.tolist(), output['scores'].cpu().tolist(), output['labels'].cpu().tolist()):
                predictions.append(dict(image_id=image_id, category_id=int(label), bbox=[x, y, max(0., x2-x), max(0., y2-y)], score=float(score)))
    tiled = getattr(loader.dataset, 'tiled_images', None)
    return merge_predictions(predictions, tiled) if tiled else predictions


def save(model, path):
    path.mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), path/'model.pt')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--sampling', choices=['natural', 'weighted'], default='weighted')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto'); p.add_argument('--resume', action='store_true')
    p.add_argument('--precision', choices=['fp32', 'amp'], default='fp32')
    args = p.parse_args(); cfg = read(args.config); root = args.root.resolve(); out = args.output.resolve(); state = None
    if min(cfg['stage_a_epochs'], cfg['stage_b_epochs'], cfg['batch_size'], cfg['accumulation'], cfg['patience']) < 1 or cfg['size'] % 32:
        raise ValueError('Positive budgets/batches/patience and a canvas divisible by 32 required')
    if args.resume:
        old = read(out/'run_config.json')
        if normalized_config(old['config']) != normalized_config(cfg) or old['sampling'] != args.sampling: raise ValueError('Resume configuration differs')
        state = torch.load(out/'last'/'training_state.pt', map_location='cpu', weights_only=False)
    else:
        out.mkdir(parents=True, exist_ok=False)
        write(out/'run_config.json', dict(config=cfg, sampling=args.sampling, checkpoint_metric='map', greater_is_better=True))
    prepared = prepared_dir(root, cfg['prepared'])
    fingerprint = prepared_fingerprint(prepared)
    if read(prepared/'split_manifest.json').get('split_fingerprint') != fingerprint: raise ValueError('Prepared split files differ from split manifest; rerun prepare_data.py')
    if args.resume and read(out/'data_identity.json')['fingerprint'] != fingerprint: raise ValueError('Prepared dataset changed since run')
    write(out/'data_identity.json', dict(fingerprint=fingerprint))
    seed_all(cfg['seed']); torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    model = build_model(cfg).to(device)
    if state: model.load_state_dict(torch.load(out/'last'/'model.pt', map_location=device))
    processor, native = TensorTargets(), cfg.get('native_pitch', False)
    train = CocoDetectionDataset(root, prepared/'train.json', processor, cfg['size'], True, cfg['augmentation'], view_mode=cfg.get('training_view_mode', 'random_augmented'),
                                 recipes=cfg.get('augmentation_recipes'), native=native)
    final_epoch = cfg.get('final_stage_b_epoch')  # a final run trains on every image, so nothing is held out
    val = None if final_epoch else CocoDetectionDataset(root, prepared/'val.json', processor, cfg['size'], native=native, tile_overlap=cfg.get('tile_overlap'))
    weights, report = sample_weights(train, cfg['sampling_power'], cfg['sampling_cap'], cfg['background_weight'])
    report['sampling'] = args.sampling; write(out/'sampling_report.json', report)
    print(f"Training: {len(train.images)} source images, {len(train)} presentations/epoch, mode={train.view_mode}, sampling={args.sampling}")
    generator = torch.Generator().manual_seed(cfg['seed'])
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator) if args.sampling == 'weighted' else None
    loader = DataLoader(train, batch_size=cfg['batch_size'], shuffle=sampler is None, sampler=sampler, num_workers=cfg['workers'], collate_fn=collate_fn, generator=generator)
    validation = DataLoader(val, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['workers'], collate_fn=collate_fn) if val else None
    amp = args.precision == 'amp' and device.type == 'cuda'
    dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    write(out/'device.json', dict(device=str(device), amp=amp, dtype=str(dtype) if amp else 'float32'))
    scaler = torch.amp.GradScaler('cuda', enabled=amp and dtype == torch.float16)
    global_best = state['global_best'] if state else -1.
    for stage, budget in [('A', cfg['stage_a_epochs']), ('B', cfg['stage_b_epochs'])]:
        if state and stage < state['stage']: continue
        resuming = state is not None and stage == state['stage']
        if stage == 'B' and not resuming and not cfg.get('fixed_schedule'):
            model.load_state_dict(torch.load(out/'stage_A_best'/'model.pt', map_location=device))
        write(out/f'trainable_stage_{stage}.json', configure_stage(model, stage, cfg['unfreeze_prefix']))
        model.train()  # ResNet BatchNorm is frozen in torchvision detection backbones
        heads = [v for n, v in model.named_parameters() if v.requires_grad and not n.startswith('backbone.body.')]
        encoder = [v for n, v in model.named_parameters() if v.requires_grad and n.startswith('backbone.body.')]
        groups = [dict(params=heads, lr=cfg['stage_a_lr'] if stage == 'A' else cfg['stage_b_head_lr'])]
        if encoder: groups.append(dict(params=encoder, lr=cfg['stage_b_encoder_lr']))
        optimizer = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
        steps = math.ceil(len(loader)/cfg['accumulation'])*budget
        scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, round(steps*cfg['warmup_ratio'])), steps)
        start, best, bad = 0, -1., 0
        if resuming:
            optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
            start, best, bad = state['epoch']+1, state['stage_best'], state['bad_epochs']
            random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng']); torch.set_rng_state(state['torch_rng']); generator.set_state(state['loader_rng'])
            if device.type == 'cuda' and state['cuda_rng'] is not None: torch.cuda.set_rng_state_all(state['cuda_rng'])
        # The budget still sets the learning-rate curve, so a final run follows the same curve as cross-validation.
        for epoch in range(start, final_epoch if final_epoch and stage == 'B' else budget):
            if bad >= cfg['patience']: break
            model.train(); optimizer.zero_grad(set_to_none=True); total_loss = 0.
            for step, batch in enumerate(loader):
                targets = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]
                group_start = (step//cfg['accumulation'])*cfg['accumulation']
                divisor = min(cfg['accumulation'], len(loader)-group_start)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                    losses = model(list(batch['pixel_values'].to(device)), targets)
                loss = sum(losses.values())
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                total_loss += float(loss.detach()); scaler.scale(loss/divisor).backward()
                if (step+1) % cfg['accumulation'] == 0 or step+1 == len(loader):
                    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad], cfg['gradient_clip'])
                    previous_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                    if scaler.get_scale() >= previous_scale: scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            predictions = predict_loader(model, validation, device) if val else []
            metrics = coco_ap(val.coco, predictions) if val else dict(map=-1.)
            score = metrics['map']
            if score is None: raise ValueError('Validation AP undefined')
            row = dict(stage=stage, epoch=epoch+1, train_loss=total_loss/len(loader), training_presentations=len(train), **metrics)
            write(out/f'metrics_{stage}_{epoch+1:03d}.json', row); print(row, flush=True)
            if cfg.get('fixed_schedule') and val: write(out/'epoch_predictions'/f'{stage}_{epoch+1:03d}.json', predictions)
            if score > best: best, bad = score, 0; save(model, out/f'stage_{stage}_best')
            else: bad += 1
            if score > global_best:
                global_best = score; save(model, out/'best'); write(out/'best'/'selection.json', dict(stage=stage, epoch=epoch+1, map=score))
            save(model, out/'last')
            torch.save(dict(stage=stage, epoch=epoch, stage_best=best, global_best=global_best, bad_epochs=bad,
                            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                            python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None, loader_rng=generator.get_state()),
                       out/'last'/'training_state.pt')
        state = None
    print(f'Final model: {out/"last"}' if final_epoch else f'Best model: {out/"best"}')


if __name__ == '__main__': main()
