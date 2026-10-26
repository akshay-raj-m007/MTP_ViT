"""Explicit validation/threshold-selection or final test evaluation."""

# Allow direct execution without installing the project package.
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sem_yolos.common import ROOT,read,write,prepared_dir
from sem_yolos.data import CocoDetectionDataset,collate_fn
from sem_yolos.runtime import device_for,load_model,predict_loader
from sem_yolos.metrics import evaluate_predictions,select_threshold


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--select-threshold',action='store_true')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=p.parse_args();torch.set_num_threads(4)
    if args.select_threshold and args.split!='val': raise ValueError('Threshold selection is validation-only')
    cfg=read(args.model/'pipeline_config.json');manifest=read(prepared_dir(args.root,cfg['prepared'])/'split_manifest.json')
    if read(args.model/'data_identity.json')['fingerprint']!=manifest['fingerprint']: raise ValueError('Model and prepared dataset identities differ')
    threshold_file=args.model/'threshold.json'
    if not args.select_threshold and not threshold_file.exists(): raise ValueError('Select a validation threshold first')
    if args.select_threshold and threshold_file.exists(): raise ValueError('Threshold already saved; use a fresh model/run for a new calibration experiment')
    args.output.mkdir(parents=True,exist_ok=False)
    device=device_for(args.device);model,processor=load_model(args.model,device)
    dataset=CocoDetectionDataset(args.root,prepared_dir(args.root,cfg['prepared'])/(args.split+'.json'),processor,cfg['size'])
    loader=DataLoader(dataset,batch_size=1,collate_fn=collate_fn,shuffle=False,num_workers=0)
    predictions=predict_loader(model,loader,device)
    if args.select_threshold:
        choice=select_threshold(dataset.coco,predictions)
        choice['data_fingerprint']=manifest['fingerprint'];write(threshold_file,choice)
    else: choice=read(threshold_file)
    if choice['data_fingerprint']!=manifest['fingerprint']: raise ValueError('Threshold belongs to different prepared data')
    metrics=evaluate_predictions(dataset.coco,predictions,choice['threshold'])
    write(args.output/'predictions.json',predictions);write(args.output/'metrics.json',metrics)
    write(args.output/'provenance.json',dict(split=args.split,model=str(args.model.resolve()),data_fingerprint=manifest['fingerprint'],threshold=choice,ap_cutoff=0.0))
    print(metrics)

if __name__=='__main__': main()
