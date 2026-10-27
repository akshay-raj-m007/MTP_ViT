import json
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageDraw
from .io import ROOT, CLASSES, write
from .dataset import SemDataset, inspect_dataset, resolve_image
from .model import create_model, configure_stage, set_training_mode, raw_forward, match_targets, decode
from .training import device_precision, autocast, seed_all


def selected_indices(dataset):
    gap = [i for i,im in enumerate(dataset.images) if
           any(dataset.mapping[a['category_id']]==1 for a in dataset.anns[im['id']])]
    selected = list(gap)
    for category in (None,0,2,3):
        for i,im in enumerate(dataset.images):
            classes = {dataset.mapping[a['category_id']] for a in dataset.anns[im['id']]}
            if (not classes if category is None else category in classes):
                if i not in selected: selected.append(i)
                break
    return selected


def preview(config, output):
    report = inspect_dataset(ROOT, ROOT/config['prepared'])
    output.mkdir(parents=True,exist_ok=False)
    original = SemDataset(ROOT,ROOT/config['prepared'],'train',config,training=False)
    augmented = SemDataset(ROOT,ROOT/config['prepared'],'train',config,training=True)
    entries=[]
    colors=['red','orange','cyan','violet']
    for index in selected_indices(original):
        source=original.images[index]
        with Image.open(resolve_image(ROOT,source['file_name'])) as opened:
            raw_source=opened.convert('RGB')
        marked=raw_source.copy();draw=ImageDraw.Draw(marked)
        for ann in original.anns[source['id']]:
            x,y,w,h=ann['bbox'];external=original.mapping[ann['category_id']]
            draw.rectangle([x,y,x+w,y+h],outline=colors[external],width=1)
            draw.text((x,max(0,y-12)),CLASSES[external],fill=colors[external])
        original_sheet=Image.new('RGB',(raw_source.width*2,raw_source.height),'white')
        original_sheet.paste(raw_source,(0,0));original_sheet.paste(marked,(raw_source.width,0))
        source_filename=f"{source['id']}_original.png"
        original_sheet.save(output/source_filename)
        for label, dataset in [('deterministic',original),('augmented',augmented)]:
            item=dataset[(index,config['seed']+index)]
            raw=Image.fromarray((item['image'].permute(1,2,0).numpy()*255).round().astype(np.uint8))
            overlay=raw.copy();draw=ImageDraw.Draw(overlay)
            for box,internal in zip(item['boxes']*config['size'],item['labels']):
                draw.rectangle(box.tolist(),outline=colors[int(internal)-1],width=1)
                draw.text((float(box[0]),max(0,float(box[1])-12)),CLASSES[int(internal)-1],fill=colors[int(internal)-1])
            sheet=Image.new('RGB',(2*config['size'],config['size']+45),'white')
            sheet.paste(raw,(0,45));sheet.paste(overlay,(config['size'],45))
            ImageDraw.Draw(sheet).text((5,5),f"train image {item['image_id']} | {label} | {len(item['labels'])} boxes\nRGB [0,1], {config['size']}px {config['resize_mode']} | unmarked / marked",fill='black')
            filename=f"{item['image_id']}_{label}.png";sheet.save(output/filename)
            entries.append(dict(image_id=item['image_id'],filename=original.images[index]['file_name'],
                                view=label,path=filename,original_preview=source_filename,boxes_normalized=item['boxes'].tolist(),
                                internal_labels=item['labels'].tolist(),metadata=item['meta']))
    write(output/'preview_manifest.json',dict(dataset=report,previews=entries,config=config))
    return dict(output=str(output),source_images=len(selected_indices(original)),transformed_previews=len(entries),
                original_previews=len(selected_indices(original)))


def smoke(config, output, device_name, precision, checkpoint_path=None):
    inspect_dataset(ROOT,ROOT/config['prepared'])
    output.mkdir(parents=True,exist_ok=False)
    device,precision=device_precision(device_name,precision)
    seed_all(config['seed']);torch.set_num_threads(4)
    model,loss_fn,report=create_model(config,checkpoint_path)
    model.to(device);loss_fn.to(device)
    dataset=SemDataset(ROOT,ROOT/config['prepared'],'train',config,training=False)
    empty=next(i for i,im in enumerate(dataset.images) if not dataset.anns[im['id']])
    multi=next(i for i,im in enumerate(dataset.images) if len(dataset.anns[im['id']])>1)
    positive=next(i for i,im in enumerate(dataset.images) if any(dataset.mapping[a['category_id']]==1 for a in dataset.anns[im['id']]))
    steps=[]
    for stage,index in [('A',positive),('A',empty),('B',multi)]:
        stage_report=configure_stage(model,stage)
        item=dataset[index];images=item['image'].unsqueeze(0).to(device)
        before={n:b.detach().clone() for n,b in model.encoder.named_buffers() if
                ('running_mean' in n or 'running_var' in n) and not(stage=='B' and n.startswith('layer_5.'))}
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5)
        scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and precision=='fp16')
        set_training_mode(model,stage);loss_fn.train();optimizer.zero_grad(set_to_none=True)
        with autocast(device,precision):pred,anchors=raw_forward(model,images)
        targets=match_targets(model,[item],anchors,device)
        assert torch.isfinite(targets['box_coordinates']).all()
        losses=loss_fn(pred,targets)
        assert torch.isfinite(losses['total_loss']) and losses['total_loss']>0
        scaler.scale(losses['total_loss']).backward();scaler.unscale_(optimizer)
        assert all(p.grad is None for p in model.encoder.parameters() if not p.requires_grad)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        gradient=sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
        assert gradient>0
        scaler.step(optimizer);scaler.update()
        after=dict(model.encoder.named_buffers())
        assert all(torch.equal(value,after[name]) for name,value in before.items())
        steps.append(dict(stage=stage,image_id=item['image_id'],targets=len(item['labels']),
                          matched_positives=int((targets['box_labels']>0).sum()),anchors=anchors.shape[1],
                          losses={k:float(v.detach()) for k,v in losses.items()},gradient_l1=gradient,
                          frozen_bn_unchanged=True,trainable_parameters=stage_report['trainable_parameters']))
    model.eval()
    val=SemDataset(ROOT,ROOT/config['prepared'],'val',config,training=False)
    item=val[0]
    with torch.no_grad(),autocast(device,precision):
        pred,anchors=raw_forward(model,item['image'].unsqueeze(0).to(device))
    detected=decode(model,pred,anchors)[0]
    assert torch.isfinite(detected.boxes).all()
    result=dict(pretrained_load=report,steps=steps,validation_decode_image_id=item['image_id'],
                decoded_boxes=len(detected.labels),device=str(device),precision=precision,
                note='Three smoke optimizer steps only; weights discarded. No test inference or model selection.')
    write(output/'smoke.json',result)
    return result
