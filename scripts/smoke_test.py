"""Real pretrained forward/backward checks; no full training or test evaluation."""

# Allow direct execution without installing the project package.
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Subset
from sem_yolos.common import ROOT,read,write
from sem_yolos.data import CocoDetectionDataset,collate_fn
from sem_yolos.runtime import seed_all,load_model,configure_stage,predict_loader
from sem_yolos.metrics import evaluate_predictions

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--size',type=int,default=512)
    p.add_argument('--config',type=Path,help='Optional config to exercise training augmentation views')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);seed_all(42)
    device=torch.device('cpu');model,processor=load_model('hustvl/yolos-tiny',device,fresh=True)
    cfg=read(a.config) if a.config else {}
    train=CocoDetectionDataset(ROOT,ROOT/'data/prepared/label_studio_v2/train.json',processor,a.size,training=bool(a.config),augmentation=cfg.get('augmentation'),view_mode=cfg.get('training_view_mode','random_augmented'),recipes=cfg.get('augmentation_recipes'))
    empty=next(i for i,im in enumerate(train.images) if not im['classes'])
    multi=next(i for i,im in enumerate(train.images) if len(train.anns_by_image[im['id']])>1)
    if train.view_mode=='original_plus_recipes':
        empty+=(1+next((i for i,r in enumerate(train.recipes) if r.get('angle')),0))*len(train.images)
        multi+=(1+next((i for i,r in enumerate(train.recipes) if r.get('angle') and (r.get('shear_x') or r.get('shear_y'))),len(train.recipes)-1))*len(train.images)
    checks=[]
    for stage,index in [('A',empty),('B',multi)]:
        report=configure_stage(model,stage);write(a.output/f'trainable_{stage}.json',report)
        batch=collate_fn([train[index]]);model.train();model.zero_grad(set_to_none=True)
        result=model(pixel_values=batch['pixel_values'],labels=batch['labels']);result.loss.backward()
        assert torch.isfinite(result.loss)
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.class_labels_classifier.parameters())
        assert all(p.grad is None for p in model.vit.embeddings.parameters())
        if stage=='B': assert any(p.grad is not None for p in model.vit.encoder.layer[-1].parameters())
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5)
        optimizer.step();checks.append(dict(stage=stage,image_id=batch['image_ids'][0],loss=float(result.loss.detach()),view=train.view_name(index)))
    val=CocoDetectionDataset(ROOT,ROOT/'data/prepared/label_studio_v2/val.json',processor,a.size)
    indices=[]
    for c in range(5):
        indices.append(next(i for i,im in enumerate(val.images) if c in (im['classes'] or [4])))
    loader=DataLoader(Subset(val,indices),batch_size=1,collate_fn=collate_fn)
    predictions=predict_loader(model,loader,device);ids={val.images[i]['id'] for i in indices}
    coco=dict(val.coco,images=[im for im in val.images if im['id'] in ids],annotations=[x for x in val.coco['annotations'] if x['image_id'] in ids])
    metrics=evaluate_predictions(coco,predictions,.5)
    write(a.output/'report.json',dict(device='cpu',size=a.size,training_view_mode=train.view_mode,presentations_per_epoch=len(train),checks=checks,validation_subset_ids=sorted(ids),metrics=metrics,
          note='Only two optimizer steps; metrics test execution, not trained-model quality. No test images used.'))
    print('SMOKE PASS',checks,'validation map',metrics['map'])
