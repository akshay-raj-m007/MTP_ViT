"""Single-GPU/CPU two-stage training with whole-validation COCO AP selection."""

# Allow direct execution without installing the project package.
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import math
import os
import random
import shutil
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,WeightedRandomSampler
from transformers import get_cosine_schedule_with_warmup
from sem_yolos.common import ROOT,read,write,prepared_dir,normalized_config,prepared_fingerprint
from sem_yolos.data import CocoDetectionDataset,collate_fn,sample_weights
from sem_yolos.runtime import seed_all,device_for,load_model,configure_stage,predict_loader
from sem_yolos.metrics import coco_ap


def save_model(model,processor,path,config):
    path.mkdir(parents=True,exist_ok=True);model.save_pretrained(path);processor.save_pretrained(path)
    write(path/'pipeline_config.json',config)


def main(default_sampling='natural'):
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--config',type=Path,default=ROOT/'configs/pipeline_config.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--sampling',choices=['natural','weighted'],default=default_sampling)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');p.add_argument('--resume',action='store_true');p.add_argument('--allow-download',action='store_true')
    args=p.parse_args();cfg=read(args.config);root=args.root.resolve();out=args.output.resolve();state=None
    if min(cfg['stage_a_epochs'],cfg['stage_b_epochs'],cfg['batch_size'],cfg['accumulation'],cfg['patience'])<1 or cfg['size']%16:
        raise ValueError('Positive budgets/batches/patience and canvas divisible by 16 required')
    if int(os.environ.get('WORLD_SIZE','1'))!=1: raise ValueError('This implementation supports one process/GPU only')
    if args.resume:
        old=read(out/'run_config.json')
        if normalized_config(old['config'])!=normalized_config(cfg) or old['sampling']!=args.sampling: raise ValueError('Resume configuration differs')
        state=torch.load(out/'last'/'training_state.pt',map_location='cpu',weights_only=False)
    else:
        out.mkdir(parents=True,exist_ok=False)
        write(out/'run_config.json',dict(config=cfg,sampling=args.sampling,checkpoint_metric='map',greater_is_better=True))
    prepared=prepared_dir(root,cfg['prepared'])
    manifest=read(prepared/'split_manifest.json')
    fingerprint=prepared_fingerprint(prepared)
    if manifest.get('split_fingerprint')!=fingerprint: raise ValueError('Prepared split files differ from split manifest; rerun prepare_data.py')
    if args.resume and read(out/'data_identity.json')['fingerprint']!=fingerprint: raise ValueError('Prepared dataset changed since run')
    write(out/'data_identity.json',dict(fingerprint=fingerprint))
    seed_all(cfg['seed']);torch.set_num_threads(min(8,os.cpu_count() or 1));device=device_for(args.device)
    source=out/'last' if state else cfg['model']
    model,processor=load_model(source,device,fresh=not bool(state),local=not args.allow_download)
    train=CocoDetectionDataset(root,prepared_dir(root,cfg['prepared'])/'train.json',processor,cfg['size'],True,cfg['augmentation'],view_mode=cfg.get('training_view_mode','random_augmented'),recipes=cfg.get('augmentation_recipes'))
    val=CocoDetectionDataset(root,prepared_dir(root,cfg['prepared'])/'val.json',processor,cfg['size'])
    weights,report=sample_weights(train,cfg['sampling_power'],cfg['sampling_cap'],cfg['background_weight'])
    report['sampling']=args.sampling
    report['expected_empty_fraction_natural']=sum(not im['classes'] for im in train.images)/len(train.images)
    report['active_expected_empty_fraction']=report['expected_empty_fraction'] if args.sampling=='weighted' else report['expected_empty_fraction_natural']
    report['original_view_guarantee']=args.sampling=='natural' and train.views_per_image>1
    print(f"Training: {len(train.images)} source images, {len(train)} presentations/epoch, mode={train.view_mode}, sampling={args.sampling}")
    write(out/'sampling_report.json',report)
    generator=torch.Generator().manual_seed(cfg['seed'])
    sampler=WeightedRandomSampler(weights,len(weights),replacement=True,generator=generator) if args.sampling=='weighted' else None
    loader=DataLoader(train,batch_size=cfg['batch_size'],shuffle=sampler is None,sampler=sampler,num_workers=cfg['workers'],collate_fn=collate_fn,generator=generator)
    validation=DataLoader(val,batch_size=cfg['batch_size'],shuffle=False,num_workers=cfg['workers'],collate_fn=collate_fn)
    amp=device.type=='cuda';dtype=torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    write(out/'device.json',dict(device=str(device),amp=amp,dtype=str(dtype) if amp else 'float32'))
    scaler=torch.amp.GradScaler('cuda',enabled=amp and dtype==torch.float16)
    global_best=state['global_best'] if state else -1.
    for stage,budget in [('A',cfg['stage_a_epochs']),('B',cfg['stage_b_epochs'])]:
        if state and stage<state['stage']: continue
        resuming=state is not None and stage==state['stage']
        if stage=='B' and not resuming:
            model,processor=load_model(out/'stage_A_best',device)
        report=configure_stage(model,stage,cfg['last_blocks']);write(out/f'trainable_stage_{stage}.json',report)
        heads=[v for n,v in model.named_parameters() if v.requires_grad and not n.startswith('vit.')]
        encoder=[v for n,v in model.named_parameters() if v.requires_grad and n.startswith('vit.')]
        groups=[dict(params=heads,lr=cfg['stage_a_lr'] if stage=='A' else cfg['stage_b_head_lr'])]
        if encoder: groups.append(dict(params=encoder,lr=cfg['stage_b_encoder_lr']))
        optimizer=torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'])
        steps=math.ceil(len(loader)/cfg['accumulation'])*budget
        scheduler=get_cosine_schedule_with_warmup(optimizer,max(1,round(steps*cfg['warmup_ratio'])),steps)
        start=0;best=-1.;bad=0
        if resuming:
            optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
            start=state['epoch']+1;best=state['stage_best'];bad=state['bad_epochs']
            random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng']);generator.set_state(state['loader_rng'])
            if amp and state['cuda_rng'] is not None: torch.cuda.set_rng_state_all(state['cuda_rng'])
        for epoch in range(start,budget):
            if bad>=cfg['patience']: break
            model.train();optimizer.zero_grad(set_to_none=True);total_loss=0.
            for step,batch in enumerate(loader):
                labels=[{k:v.to(device) if torch.is_tensor(v) else v for k,v in t.items()} for t in batch['labels']]
                group_start=(step//cfg['accumulation'])*cfg['accumulation']
                divisor=min(cfg['accumulation'],len(loader)-group_start)
                with torch.autocast(device_type=device.type,dtype=dtype,enabled=amp):
                    loss=model(pixel_values=batch['pixel_values'].to(device),labels=labels).loss
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                total_loss+=float(loss.detach());scaler.scale(loss/divisor).backward()
                if (step+1)%cfg['accumulation']==0 or step+1==len(loader):
                    scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
                    previous_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
                    if scaler.get_scale()>=previous_scale: scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            predictions=predict_loader(model,validation,device);metrics=coco_ap(val.coco,predictions)
            score=metrics['map']
            if score is None: raise ValueError('Validation AP undefined')
            row=dict(stage=stage,epoch=epoch+1,train_loss=total_loss/len(loader),training_presentations=len(train),optimizer_update_opportunities=math.ceil(len(loader)/cfg['accumulation']),**metrics)
            write(out/f'metrics_{stage}_{epoch+1:03d}.json',row);print(row)
            if score>best:
                best=score;bad=0;save_model(model,processor,out/f'stage_{stage}_best',cfg)
            else: bad+=1
            if score>global_best:
                global_best=score;save_model(model,processor,out/'best',cfg)
                write(out/'best'/'selection.json',dict(stage=stage,epoch=epoch+1,map=score))
                write(out/'best'/'data_identity.json',dict(fingerprint=fingerprint))
            save_model(model,processor,out/'last',cfg)
            torch.save(dict(stage=stage,epoch=epoch,stage_best=best,global_best=global_best,bad_epochs=bad,
                            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all() if amp else None,loader_rng=generator.get_state()),out/'last'/'training_state.pt')
        state=None
    print(f'Best model: {out / "best"}. Select confidence threshold on validation before test/inference.')

if __name__=='__main__': main()
