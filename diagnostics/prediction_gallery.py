"""Offline validation/test prediction gallery. Test visualization is explicitly opt-in here.

Separate from sem_defects' validation-only diagnostics. No training or metric/threshold selection.
"""
import sys
sys.dont_write_bytecode=True
import argparse
import hashlib
import html
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUNS={
 'yolos_unweighted':('yolos','outputs/runs/unweighted_gap_resplit_01'),
 'yolos_weighted':('yolos','outputs/runs/weighted_gap_resplit_01'),
 'ssd_unweighted':('ssd','outputs/mobilevitv2_ssd/natural_seed42_retry_20261027_230508'),
 'ssd_weighted':('ssd','outputs/mobilevitv2_ssd/weighted_seed42')}
COLORS={'bridge':'#e53935','gap':'#ff9800','line_collapse':'#2979ff','multi_bridge':'#ab47bc'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def fresh(path):
    p=(ROOT/path).resolve()
    if not p.is_relative_to((ROOT/'outputs/prediction_gallery').resolve()):raise ValueError('Output must be under outputs/prediction_gallery/')
    p.mkdir(parents=True,exist_ok=False);return p


def settings(model):
    family,relative=RUNS[model];run=ROOT/relative
    if family=='yolos':
        cfg=read(run/'best/pipeline_config.json');labels=read(run/'best/config.json')['id2label']
        threshold=read(run/'best/threshold.json');fingerprint=read(run/'best/data_identity.json')['fingerprint']
        if threshold['data_fingerprint']!=fingerprint:raise ValueError('YOLOS threshold/data mismatch')
        t=threshold['threshold'];threshold_source='validation-selected operating threshold'
    else:
        saved=read(run/'run_config.json');cfg=saved['config'];labels=saved['external_labels'];fingerprint=saved['data_fingerprint']
        t=read(run/'best_validation_metrics.json')['class_agnostic_localization']['confidence_threshold']
        threshold_source='existing fixed localization diagnostic threshold; not calibrated'
    return family,run,cfg,{int(k):v for k,v in labels.items()},fingerprint,t,threshold_source


def capture_ssd_test(args):
    import torch
    from PIL import Image
    sys.path.insert(0,str(ROOT/'experiments/mobilevitv2_ssd'))
    from sem_ssd.model import create_model,raw_forward,decode
    from sem_ssd.preprocessing import transform,inverse
    device=torch.device(args.device)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    for key in ('ssd_unweighted','ssd_weighted'):
        if not (ROOT/RUNS[key][1]/'best.pt').is_file():raise FileNotFoundError(f'Trained checkpoint required: {ROOT/RUNS[key][1]/"best.pt"}')
    out=fresh(args.output);torch.set_num_threads(4)
    for key in ('ssd_unweighted','ssd_weighted'):
        _,run,cfg,labels,fingerprint,threshold,_=settings(key)
        saved=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        selection=read(run/'best_selection.json')
        if saved['config']!=cfg or saved['data_fingerprint']!=fingerprint or saved['stage']!=selection['stage'] or saved['epoch']!=selection['epoch']:
            raise ValueError('Checkpoint differs from validation-selected best model')
        if read(run/'run_config.json')['precision']!='fp32':raise ValueError('This export requires the recorded fp32 baseline')
        manifest=read(ROOT/cfg['prepared']/'split_manifest.json')
        if manifest['split_fingerprint']!=fingerprint:raise ValueError('Dataset identity mismatch')
        model,_,_=create_model(cfg,pretrained=False);model.load_state_dict(saved['model'],strict=True);model.to(device).eval()
        splits=['test'] if args.command=='capture-ssd-test' else ['val','test'] if args.split=='both' else [args.split]
        for split in splits:
            annotation_file=ROOT/cfg['prepared']/f'{split}.json';data=read(annotation_file)
            predictions=[]
            with torch.inference_mode():
                for info in data['images']:
                    path=ROOT/info['file_name']
                    with Image.open(path) as im:
                        if im.size!=(info['width'],info['height']):raise ValueError('Image dimensions differ')
                        tensor,_,meta,_=transform(im,[],cfg['size'],cfg['resize_mode'])
                    pred,anchors=raw_forward(model,tensor.unsqueeze(0).to(device));result=decode(model,pred,anchors)[0]
                    boxes=inverse(result.boxes.cpu(),meta)
                    for box,label,score in zip(boxes.tolist(),result.labels.tolist(),result.scores.tolist()):
                        x,y,x2,y2=box
                        if not all(math.isfinite(v) for v in box+[score]):raise ValueError('Nonfinite prediction')
                        if x2>x and y2>y:predictions.append(dict(image_id=info['id'],category_id=int(label)-1,bbox=[x,y,x2-x,y2-y],score=float(score)))
                    print(f'{key}: {split} image {info["id"]}',flush=True)
            write(out/f'{key}_{split}.json',predictions)
            write(out/f'{key}_{split}_provenance.json',dict(split=split,model=key,checkpoint=str(run/'best.pt'),checkpoint_sha256=sha(run/'best.pt'),
                data_fingerprint=fingerprint,annotation_sha256=sha(annotation_file),precision='fp32',device=str(device),
                image_ids=[im['id'] for im in data['images']],stage=saved['stage'],epoch=saved['epoch'],threshold=threshold,
                note='Inference for visualization only; no test metrics or parameter selection.'))
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
    print(out)


def build(args):
    from PIL import Image
    out=fresh(args.output);(out/'images').mkdir()
    sections=[];provenance={};split_data={};copied={};counts={}
    for split in ('val','test'):
        rows=[];model_data={};available={}
        for model in RUNS:
            family,run,cfg,labels,fingerprint,threshold,threshold_source=settings(model)
            ann_path=ROOT/cfg['prepared']/f'{split}.json';data=read(ann_path)
            names={c['id']:c['name'] for c in data['categories']}
            if set(names.values())!=set(labels.values()):raise ValueError('Class mappings differ')
            if read(ROOT/cfg['prepared']/'split_manifest.json')['split_fingerprint']!=fingerprint:raise ValueError('Split identity mismatch')
            if split in split_data and split_data[split]!=sha(ann_path):raise ValueError('Models use different splits')
            split_data[split]=sha(ann_path)
            if family=='yolos':
                folder=run/('validation' if split=='val' else 'final_test');path=folder/'predictions.json'
                meta=read(folder/'provenance.json')
                if meta['split']!=split or meta['data_fingerprint']!=fingerprint:raise ValueError('Saved prediction provenance mismatch')
            else:
                exported=(ROOT/args.ssd_test_dir/f'{model}_{split}.json') if args.ssd_test_dir else None
                path=exported if exported and exported.is_file() else run/'best_validation_predictions.json' if split=='val' else None
                if exported and exported.is_file():
                    meta=read(exported.with_name(f'{model}_{split}_provenance.json'))
                    if meta['split']!=split or meta['model']!=model or meta['data_fingerprint']!=fingerprint or meta['annotation_sha256']!=sha(ann_path):
                        raise ValueError('SSD export provenance mismatch')
                    selection=read(run/'best_selection.json')
                    if meta['stage']!=selection['stage'] or meta['epoch']!=selection['epoch']:raise ValueError('Predictions from a different checkpoint selection')
            exists=path is not None and path.is_file();pred=read(path) if exists else None
            ids={im['id'] for im in data['images']}
            if pred is not None:
                for p in pred:
                    if p['image_id'] not in ids or p['category_id'] not in labels:raise ValueError('Invalid prediction reference')
                    if not all(math.isfinite(v) for v in p['bbox']+[p['score']]) or min(p['bbox'][2:])<0:raise ValueError('Invalid prediction coordinates')
            provenance[f'{split}/{model}']=dict(predictions=str(path) if path else None,available=exists,
                sha256=sha(path) if exists else None,threshold=threshold,threshold_source=threshold_source,
                data_fingerprint=fingerprint,annotation_sha256=sha(ann_path),labels=labels)
            model_data[model]=(pred,labels,threshold);available[model]=exists
        counts[split]=dict(images=len(data['images']),models_available=available)
        for info in data['images']:
            image_id=info['id'];path=ROOT/info['file_name']
            with Image.open(path) as im:
                if im.size!=(info['width'],info['height']):raise ValueError('Image size mismatch')
            key=f'{split}_{image_id}';name=f'images/{key}{path.suffix.lower()}'
            shutil.copyfile(path,out/name);copied[key]=name
            row=f'<article><h3>{html.escape(Path(info["file_name"]).name)} · image {image_id}</h3><a href="{name}" target="_blank">Open original image</a><div class="cards">'
            for model,(pred,labels,threshold) in model_data.items():
                row+=f'<div class="card"><h4>{model.replace("_"," ")}</h4><p>Confidence ≥ {threshold:.2f}</p>'
                if pred is None:
                    row+='<div class="missing">Predictions have not been supplied for this view.<br>Use the SSD capture command and provide its export directory.<br>This does not mean “no detection”.</div></div>';continue
                chosen=sorted([p for p in pred if p['image_id']==image_id and p['score']>=threshold],key=lambda p:p['score'],reverse=True)
                width,height=info['width'],info['height']
                svg=f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{model} predictions for image {image_id}"><image href="{name}" width="{width}" height="{height}"/>'
                for i,p in enumerate(chosen,1):
                    x,y,w,h=p['bbox'];cls=labels[p['category_id']];color=COLORS[cls]
                    svg+=f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{color}" stroke-width="2" vector-effect="non-scaling-stroke"><title>#{i}: {cls}, confidence {p["score"]:.5f}</title></rect>'
                svg+='</svg>'
                row+=svg+f'<p class="count">{len(chosen)} predicted boxes at the displayed threshold</p>'
                if not chosen:row+='<p>No prediction above this threshold; this is not a guarantee the image is clean.</p>'
                else:
                    row+='<details><summary>Classes, confidence and coordinates</summary><ol>'
                    for p in chosen:
                        cls=labels[p['category_id']];box=', '.join(f'{v:.1f}' for v in p['bbox'])
                        row+=f'<li><span style="color:{COLORS[cls]}">{cls}</span>: {p["score"]:.4f}; xywh [{box}]</li>'
                    row+='</ol></details>'
                row+='</div>'
            rows.append(row+'</div></article>')
        sections.append(f'<section id="{split}"><h2>{"Validation" if split=="val" else "Test"} — {len(data["images"])} images</h2>'+''.join(rows)+'</section>')
    legend=' '.join(f'<span style="color:{color};font-weight:bold">■ {name}</span>' for name,color in COLORS.items())
    body='''<!doctype html><meta charset="utf-8"><title>SEM predicted bounding boxes</title>
<style>body{font:16px/1.5 system-ui;margin:25px;background:#f4f7fb;color:#152940}nav{position:sticky;top:0;background:#fff;padding:12px;z-index:3}nav a{margin-right:24px}article{background:white;margin:25px 0;padding:20px;border-radius:10px}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.card{border:1px solid #ccd5e0;padding:14px;min-width:0}svg{width:100%;max-height:650px;background:#eee}.missing{padding:50px 20px;background:#fff0cb}li{overflow-wrap:anywhere}a{color:#145fb0}.intro{padding:18px;background:#fff}h4{font-size:19px;margin:0}.count{font-weight:600}@media(max-width:800px){.cards{grid-template-columns:1fr}}@media(min-width:1900px){.cards{grid-template-columns:repeat(4,minmax(0,1fr))}}</style>
<h1>SEM predicted bounding boxes — four models</h1><nav><a href="#val">Validation images</a><a href="#test">Test images</a><a href="manifest.json">Provenance</a></nav>
<div class="intro"><p>Prediction boxes only, in original-image coordinates. Ground-truth boxes are not drawn. Hover a box for its class/confidence; expand the list for coordinates. All saved predictions above each model’s existing display threshold are shown. No threshold tuning or new performance metrics.</p>
<p>YOLOS uses its validation-selected thresholds. SSD uses its existing fixed 0.30 diagnostic threshold, not a calibrated deployment threshold. Confidence values are not calibrated across models. Test views are separated from validation; do not use them to select settings.</p>'''+legend+'</div>'+''.join(sections)
    (out/'index.html').write_text(body,encoding='utf-8')
    write(out/'manifest.json',dict(created=datetime.now().isoformat(),purpose='User-requested validation/test prediction visualization',
        counts=counts,sources=provenance,source_script_sha256=sha(__file__),note='Existing files are read-only. Missing predictions are marked unavailable, never treated as empty.'))
    print(json.dumps(counts,indent=2));print(f'Open: {out/"index.html"}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['build','capture-ssd-test','capture-ssd'])
    parser.add_argument('--output',type=Path,default=Path('outputs/prediction_gallery')/datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--ssd-test-dir','--ssd-predictions-dir',dest='ssd_test_dir',type=Path,help='Directory produced by SSD capture')
    parser.add_argument('--split',choices=['val','test','both'],default='both',help='Splits for capture-ssd; capture-ssd-test always uses test')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    capture_ssd_test(args) if args.command in ('capture-ssd-test','capture-ssd') else build(args)
