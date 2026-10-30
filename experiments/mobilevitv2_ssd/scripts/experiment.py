"""Isolated experiment commands; full training requires an explicit train/overfit command."""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sem_ssd.io import ROOT, EXPERIMENT, CACHE, read, write, output_path, isolated_environment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['setup','inspect','check-setup','preview','smoke','train','overfit','validate','test'])
    p.add_argument('--config',type=Path,default=EXPERIMENT/'configs/natural.json')
    p.add_argument('--prepared',type=str)
    p.add_argument('--output',type=Path)
    p.add_argument('--checkpoint',type=Path,help='Verified complete official COCO checkpoint')
    p.add_argument('--model',type=Path,help='Experiment best.pt for validation only')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='cpu')
    p.add_argument('--precision',choices=['auto','fp32','fp16','bf16'],default='fp32')
    p.add_argument('--seed',type=int)
    p.add_argument('--size',type=int)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--early-stopping',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--skip-weights',action='store_true')
    p.add_argument('--subset-size',type=int,default=4)
    p.add_argument('--epochs',type=int,default=30,help='Stage B epochs for optional subset overfit check')
    args=p.parse_args();isolated_environment()
    if args.command=='setup':
        from sem_ssd.upstream import setup
        print(json.dumps(setup(args.skip_weights),indent=2));return
    config=read(args.config)
    if args.prepared:config['prepared']=args.prepared
    if args.seed is not None:config['seed']=args.seed
    if args.size is not None:config['size']=args.size
    if args.early_stopping is not None:config['early_stopping']=args.early_stopping
    if args.command in ('train','overfit') and not args.output:p.error('--output is required for training')
    out=output_path(args.output or CACHE/(args.command+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]))
    if args.command=='inspect':
        from sem_ssd.dataset import inspect_dataset,SemDataset
        from sem_ssd.sampling import weights_and_report
        report=inspect_dataset(ROOT,ROOT/config['prepared'])
        report['sampling']=weights_and_report(SemDataset(ROOT,ROOT/config['prepared'],'train',config),config)[1]
        out.mkdir(parents=True,exist_ok=False);write(out/'dataset_report.json',report)
        print(json.dumps(report,indent=2));print(out);return
    if args.command=='check-setup':
        import torch
        from sem_ssd.model import create_model,raw_forward
        from sem_ssd.training import device_precision,autocast
        device,precision=device_precision(args.device,args.precision)
        torch.set_num_threads(4)
        model,_,report=create_model(config,args.checkpoint);model.to(device).eval()
        with torch.no_grad(),autocast(device,precision):
            pred,anchors=raw_forward(model,torch.zeros(1,3,config['size'],config['size'],device=device))
        report.update(input_size=config['size'],anchors=anchors.shape[1],scores_shape=list(pred['scores'].shape),
                      trainable_parameters=sum(x.numel() for x in model.parameters() if x.requires_grad))
        out.mkdir(parents=True,exist_ok=False);write(out/'setup_report.json',report)
        print(json.dumps(report,indent=2));print(out);return
    if args.command=='preview':
        from sem_ssd.verification import preview
        print(json.dumps(preview(config,out),indent=2));return
    if args.command=='smoke':
        from sem_ssd.verification import smoke
        print(json.dumps(smoke(config,out,args.device,args.precision,args.checkpoint),indent=2));return
    if args.command in ('validate','test'):
        # 'test' scores the held-out split once with the validation-selected best.pt; it selects nothing.
        if not args.model:p.error('--model is required')
        split='test' if args.command=='test' else 'val'
        import torch
        from sem_ssd.dataset import inspect_dataset,SemDataset
        from sem_ssd.model import create_model
        from sem_ssd.training import validate,device_precision
        saved=torch.load(args.model,map_location='cpu',weights_only=False)
        config=saved['config']
        report=inspect_dataset(ROOT,ROOT/config['prepared'])
        if saved['data_fingerprint']!=report['fingerprint']:raise ValueError('Checkpoint/data mismatch')
        model,_,_=create_model(config,pretrained=False);model.load_state_dict(saved['model'],strict=True)
        device,precision=device_precision(args.device,args.precision);model.to(device)
        metrics,predictions=validate(model,SemDataset(ROOT,ROOT/config['prepared'],split,config,final_test=split=='test',tiled=True),device,precision)
        out.mkdir(parents=True,exist_ok=False);write(out/'metrics.json',metrics);write(out/'predictions.json',predictions)
        write(out/'provenance.json',dict(model=str(args.model),split=split,data_fingerprint=report['fingerprint'],config=config,precision=precision))
        print(json.dumps(metrics,indent=2));return
    from sem_ssd.training import train
    subset=None
    if args.command=='overfit':
        if args.subset_size<1 or args.epochs<1:p.error('Positive subset size and epochs required')
        subset=args.subset_size
        config.update(sampling='natural',early_stopping=False,stage_a_epochs=5,stage_b_epochs=args.epochs,accumulation=1)
        config['augmentation']={k:0. for k in config['augmentation']}
        config['training_view_mode']='random_augmented'  # overfit check memorises unaugmented images
    train(config,out,args.device,args.precision,args.resume,args.checkpoint,subset)


if __name__=='__main__':
    main()
