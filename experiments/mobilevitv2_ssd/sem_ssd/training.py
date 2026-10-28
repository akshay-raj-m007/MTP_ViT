"""Two-stage fine-tuning; validation-only selection and epoch-boundary resume."""
import contextlib
import math
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import SemDataset, collate, inspect_dataset
from .io import ROOT, REVISION, CHECKPOINT_SHA256, digest, output_path, write
from .metrics import coco_ap, localization_recall
from .model import create_model, configure_stage, set_training_mode, raw_forward, match_targets, decode
from .preprocessing import inverse
from .sampling import epoch_keys, weights_and_report


def device_precision(device_name, requested):
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if requested == 'auto':
        requested = ('bf16' if torch.cuda.is_bf16_supported() else 'fp16') if device.type=='cuda' else 'fp32'
    if requested != 'fp32' and device.type != 'cuda':
        raise ValueError('CPU verification uses fp32; fp16/bf16 require CUDA in this experiment')
    if requested == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support bf16')
    return device, requested


def autocast(device, precision):
    return (contextlib.nullcontext() if precision=='fp32' else
            torch.autocast(device.type, dtype={'fp16':torch.float16,'bf16':torch.bfloat16}[precision]))


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint(path, data):
    temporary = path.with_name(path.name+'.tmp')
    torch.save(data, temporary)
    temporary.replace(path)


def accumulation_divisor(step, batches, accumulation):
    return min(accumulation, batches-(step//accumulation)*accumulation)


@torch.no_grad()
def validate(model, dataset, device, precision, workers=0):
    model.eval()
    predictions = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers, collate_fn=collate)
    for images, targets in loader:
        with autocast(device, precision):
            pred, anchors = raw_forward(model, images.to(device))
        decoded = decode(model, pred, anchors)
        for result, target in zip(decoded, targets):
            boxes = inverse(result.boxes.detach().cpu(), target['meta'])
            for box, label, score in zip(boxes.tolist(), result.labels.tolist(), result.scores.tolist()):
                x,y,x2,y2 = box
                if not all(math.isfinite(v) for v in box+[score]):
                    raise RuntimeError('Nonfinite decoded prediction')
                if x2 <= x or y2 <= y:
                    continue  # decoded predictions outside the source content; never GT
                predictions.append(dict(image_id=target['image_id'], category_id=int(label)-1,
                                        bbox=[x,y,x2-x,y2-y], score=float(score)))
    ground_truth=dataset.canonical_coco()
    metrics=coco_ap(ground_truth, predictions)
    metrics['class_agnostic_localization']=localization_recall(ground_truth,predictions)
    return metrics, predictions


def train(config, output, device_name='cuda', precision='fp32', resume=False,
          checkpoint_path=None, subset_size=None):
    root = ROOT
    prepared = root / config['prepared']
    data_report = inspect_dataset(root, prepared)
    device, precision = device_precision(device_name, precision)
    if config['workers'] < 0 or min(config['batch_size'],config['accumulation'],config['stage_a_epochs'],config['stage_b_epochs'],config['patience'],config['empty_negatives']) < 1:
        raise ValueError('Positive training budgets, accumulation, patience and empty negatives required')
    if config['size'] < 256 or config['size'] % 32:
        raise ValueError('SSD size must be a multiple of 32 and at least 256')
    out = output_path(output)
    identity = dict(config=config, precision=precision, device_type=device.type,
                    data_fingerprint=data_report['fingerprint'], upstream_revision=REVISION,
                    checkpoint_sha256=CHECKPOINT_SHA256, subset_size=subset_size)
    identity_hash = digest(identity)
    state = None
    if resume:
        state = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
        if state['identity_hash'] != identity_hash:
            raise ValueError('Resume requires identical config, dataset, upstream, precision, device and subset')
    else:
        out.mkdir(parents=True, exist_ok=False)
    seed_all(config['seed'])
    torch.set_num_threads(4)
    model, loss_fn, load_report = create_model(config, checkpoint_path, pretrained=state is None)
    if state:
        model.load_state_dict(state['model'], strict=True)
    model.to(device); loss_fn.to(device)
    if state is None:
        write(out/'pretrained_load.json', load_report)
    indices = None
    full_train = SemDataset(root, prepared, 'train', config, training=True)
    if subset_size:
        # Deterministic small subset, starting with a gap and an empty image if available.
        chosen = []
        for wanted in (1, None):
            for i, im in enumerate(full_train.images):
                classes = {full_train.mapping[a['category_id']] for a in full_train.anns[im['id']]}
                if (not classes if wanted is None else wanted in classes):
                    chosen.append(i); break
        chosen += [i for i in range(len(full_train.images)) if i not in chosen]
        indices = chosen[:min(subset_size,len(full_train.images))]
    train_data = SemDataset(root, prepared, 'train', config, training=True, indices=indices)
    val_data = (SemDataset(root, prepared, 'train', config, training=False, indices=indices) if subset_size else
                SemDataset(root, prepared, 'val', config))
    _, sampling_report = weights_and_report(train_data, config)
    write(out/'sampling_report.json', sampling_report)
    write(out/'dataset_report.json', data_report)
    write(out/'run_config.json', dict(**identity, identity_hash=identity_hash,
          selection_split='training_subset_debug_only' if subset_size else 'val',
          external_labels={i:c for i,c in enumerate(['bridge','gap','line_collapse','multi_bridge'])}))
    import torchvision
    runtime = dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
                   torchvision=torchvision.__version__, numpy=np.__version__, device=str(device), precision=precision,
                   gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                   cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    write(out/('runtime_resume.json' if resume else 'runtime.json'), runtime)
    sampling_rng = torch.Generator().manual_seed(config['seed']+101)
    augmentation_rng = torch.Generator().manual_seed(config['seed']+202)
    loader_rng = torch.Generator().manual_seed(config['seed']+303)
    if state:
        random.setstate(state['rng']['python']); np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch'])
        if device.type=='cuda': torch.cuda.set_rng_state_all(state['rng']['cuda'])
        sampling_rng.set_state(state['rng']['sampling']); augmentation_rng.set_state(state['rng']['augmentation'])
        loader_rng.set_state(state['rng']['loader'])
    global_best = state['global_best'] if state else -1.
    total_updates = state['total_updates'] if state else 0
    total_presentations = state['total_presentations'] if state else 0
    for stage, budget in [('A',config['stage_a_epochs']),('B',config['stage_b_epochs'])]:
        if state and stage < state['stage']:
            continue
        continuing = state is not None and stage == state['stage']
        if stage=='B' and not continuing:
            best_a = torch.load(out/'stage_A_best.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(best_a['model'], strict=True)
        stage_report = configure_stage(model, stage)
        write(out/f'trainable_stage_{stage}.json', stage_report)
        head = [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('encoder.')]
        encoder = [p for n,p in model.named_parameters() if p.requires_grad and n.startswith('encoder.')]
        groups = [dict(params=head, lr=config['stage_a_lr'] if stage=='A' else config['stage_b_head_lr'])]
        if encoder: groups.append(dict(params=encoder, lr=config['stage_b_encoder_lr']))
        optimizer = torch.optim.AdamW(groups, weight_decay=config['weight_decay'])
        batches = math.ceil(len(train_data)/config['batch_size'])
        total_steps = math.ceil(batches/config['accumulation'])*budget
        warmup = max(1, round(total_steps*config['warmup_ratio']))
        def lr_factor(step):
            if step < warmup: return (step+1)/warmup
            return .5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,total_steps-warmup))))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        scaler = torch.amp.GradScaler('cuda', enabled=device.type=='cuda' and precision=='fp16')
        start, stage_best, bad = 0, -1., 0
        if continuing:
            optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
            scaler.load_state_dict(state['scaler'])
            start, stage_best, bad = state['epoch'], state['stage_best'], state['bad_epochs']
        for epoch in range(start, budget):
            if config['early_stopping'] and bad >= config['patience']:
                break
            started = time.perf_counter()
            if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
            keys = epoch_keys(train_data, config, sampling_rng, augmentation_rng)
            loader = DataLoader(train_data, batch_size=config['batch_size'], sampler=keys,
                num_workers=config['workers'], collate_fn=collate, generator=loader_rng,
                pin_memory=device.type=='cuda')
            set_training_mode(model, stage); loss_fn.train(); optimizer.zero_grad(set_to_none=True)
            totals = dict(total_loss=0., reg_loss=0., cls_loss=0., empty_background_loss=0.)
            updates = 0
            for step, (images, targets) in enumerate(loader):
                images = images.to(device)
                with autocast(device, precision):
                    pred, anchors = raw_forward(model, images)
                matched = match_targets(model, targets, anchors, device)
                losses = loss_fn(pred, matched)
                if not torch.isfinite(losses['total_loss']): raise RuntimeError('Nonfinite training loss')
                divisor = accumulation_divisor(step, len(loader), config['accumulation'])
                scaler.scale(losses['total_loss']/divisor).backward()
                for k in totals: totals[k] += float(losses[k].detach())
                total_presentations += len(targets)
                if (step+1)%config['accumulation']==0 or step+1==len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                    before_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                    if scaler.get_scale() >= before_scale:
                        scheduler.step(); updates += 1; total_updates += 1
                    optimizer.zero_grad(set_to_none=True)
            metrics, predictions = validate(model, val_data, device, precision, config['workers'])
            score = metrics['map']
            if score is None: raise ValueError('Selection AP undefined')
            row = dict(stage=stage, epoch=epoch+1, **metrics,
                       losses={k:v/len(loader) for k,v in totals.items()},
                       presentations=len(train_data), optimizer_updates=updates,
                       optimizer_update_opportunities=math.ceil(len(loader)/config['accumulation']),
                       total_updates=total_updates, total_presentations=total_presentations,
                       epoch_seconds=time.perf_counter()-started,
                       peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
            write(out/f'metrics_{stage}_{epoch+1:03d}.json', row)
            print(row, flush=True)
            selected = dict(model=model.state_dict(), config=config, data_fingerprint=data_report['fingerprint'],
                            identity_hash=identity_hash, stage=stage, epoch=epoch+1, map=score,
                            upstream_revision=REVISION, checkpoint_sha256=CHECKPOINT_SHA256)
            if score > stage_best:
                stage_best, bad = score, 0
                checkpoint(out/f'stage_{stage}_best.pt', selected)
            else: bad += 1
            if score > global_best:
                global_best = score
                checkpoint(out/'best.pt', selected)
                write(out/'best_validation_metrics.json', metrics)
                write(out/'best_validation_predictions.json', predictions)
                write(out/'best_selection.json', dict(stage=stage,epoch=epoch+1,map=score,
                    data_fingerprint=data_report['fingerprint'], score_floor=config['score_floor'],
                    nms_iou=config['nms_iou'], pre_nms_topk=config['pre_nms_topk'],max_detections=config['max_detections']))
            checkpoint(out/'last.pt', dict(**selected, optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(), precision=precision, global_best=global_best,stage_best=stage_best,bad_epochs=bad,
                total_updates=total_updates,total_presentations=total_presentations,
                rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                         cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                         sampling=sampling_rng.get_state(),augmentation=augmentation_rng.get_state(),loader=loader_rng.get_state())))
        state = None
    print(f'Finished. Validation-selected checkpoint: {out / "best.pt"}', flush=True)
