import random
import numpy as np
import torch
from transformers import YolosForObjectDetection,YolosImageProcessor
from .common import CLASSES
from .data import inverse_boxes
from .windows import merge_predictions


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def device_for(name='auto'):
    if name=='auto': name='cuda' if torch.cuda.is_available() else 'cpu'
    if name=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    return torch.device(name)


def load_model(source,device,fresh=False,local=True):
    kw=dict(local_files_only=local,attn_implementation='eager')
    if fresh: kw.update(num_labels=4,id2label=dict(enumerate(CLASSES)),label2id={n:i for i,n in enumerate(CLASSES)},ignore_mismatched_sizes=True)
    model=YolosForObjectDetection.from_pretrained(source,**kw).to(device)
    if model.config.num_labels!=4 or [model.config.id2label[i] for i in range(4)]!=CLASSES:
        raise ValueError('Expected new four-class canonical model; old checkpoints are incompatible')
    processor=YolosImageProcessor.from_pretrained(source,local_files_only=local)
    processor.do_resize=False;processor.do_pad=False
    return model,processor


def decode(outputs,metas,image_ids):
    probabilities=outputs.logits.softmax(-1)[...,:-1]
    scores,labels=probabilities.max(-1)
    result=[]
    for boxes,sc,lab,meta,img_id in zip(outputs.pred_boxes,scores,labels,metas,image_ids):
        cx,cy,w,h=boxes.detach().cpu().unbind(-1)
        xyxy=torch.stack((cx-w/2,cy-h/2,cx+w/2,cy+h/2),-1)*meta['size']
        xyxy=inverse_boxes(xyxy,meta)
        for b,s,c in zip(xyxy,sc.detach().cpu(),lab.detach().cpu()):
            x,y,x2,y2=b.tolist()
            result.append(dict(image_id=img_id,category_id=int(c),bbox=[x,y,max(0.,x2-x),max(0.,y2-y)],score=float(s)))
    return result


@torch.no_grad()
def predict_loader(model,loader,device):
    model.eval();predictions=[]
    for batch in loader:
        outputs=model(pixel_values=batch['pixel_values'].to(device))
        predictions.extend(decode(outputs,batch['meta'],batch['image_ids']))
    tiled=getattr(loader.dataset,'tiled_images',None)
    return merge_predictions(predictions,tiled) if tiled else predictions


def configure_stage(model,stage,last_blocks=2):
    for p in model.parameters(): p.requires_grad=False
    for module in [model.class_labels_classifier,model.bbox_predictor]:
        for p in module.parameters(): p.requires_grad=True
    if stage=='B':
        if not 1<=last_blocks<=len(model.vit.encoder.layer): raise ValueError('Invalid last_blocks')
        for module in list(model.vit.encoder.layer)[-last_blocks:]+[model.vit.layernorm]:
            for p in module.parameters(): p.requires_grad=True
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    report=dict(stage=stage,names=names,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),total_parameters=sum(p.numel() for p in model.parameters()))
    print(report)
    return report
