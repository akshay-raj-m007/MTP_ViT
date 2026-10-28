"""Offline PNG/HTML reports. Images retain fixed RGB intensity; no display auto-contrast."""
import html
import math
import os
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from core import image_path,category_mapping,read,write,save_review

COLORS={'gap':'#ffb000','bridge':'#f15b5b','line_collapse':'#408cff','multi_bridge':'#df70ff'}


def font(size=15):
    for name in ('DejaVuSans.ttf','arial.ttf'):
        try: return ImageFont.truetype(name,size)
        except OSError: pass
    return ImageFont.load_default()


def overlay(image,annotations,predictions=()):
    canvas=image.copy();draw=ImageDraw.Draw(canvas)
    for a in annotations:
        x,y,w,h=a['bbox']; color=COLORS[a['class_name']]
        draw.rectangle((x,y,x+w,y+h),outline=color,width=1)
    for p in predictions:
        x,y,w,h=p['bbox']
        draw.rectangle((x,y,x+w,y+h),outline='#00e5c3',width=1)
    return canvas


def crop_box(image,ann):
    x,y,w,h=ann['bbox'];margin=max(12,w*.75,h*.75)
    return (max(0,math.floor(x-margin)),max(0,math.floor(y-margin)),
            min(image.width,math.ceil(x+w+margin)),min(image.height,math.ceil(y+h+margin)))


def panel(tiles,path):
    import textwrap
    width=520;height=590;cols=2
    sheet=Image.new('RGB',(cols*width,math.ceil(len(tiles)/cols)*height),'#f1f4f8');draw=ImageDraw.Draw(sheet)
    for i,(title,image) in enumerate(tiles):
        x=(i%cols)*width;y=(i//cols)*height
        scale=min((width-24)/image.width,(height-140)/image.height)
        thumb=image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))),Image.Resampling.NEAREST)
        draw.multiline_text((x+12,y+12),'\n'.join(textwrap.wrap(title,60)),fill='#102030',font=font(14),spacing=4)
        draw.text((x+12,y+103),f'View scale {scale:.2f}x, nearest-neighbor; no contrast adjustment',fill='#35465a',font=font(12))
        sheet.paste(thumb,(x+12,y+130))
    sheet.save(path)


def dataset_gallery(datasets,image_root,output,report):
    sections=[];review=[]
    for split,data in datasets.items():
        mapping=category_mapping(data['categories']);images={im['id']:im for im in data['images']}
        annotations=[dict(a,class_name=mapping[a['category_id']]) for a in data['annotations']]
        tiles=[]
        for ann in annotations:
            if ann['class_name']!='gap':continue
            info=images[ann['image_id']]
            with Image.open(image_path(image_root,info)) as im: raw=im.convert('RGB')
            anns=[a for a in annotations if a['image_id']==info['id']]
            region=crop_box(raw,ann);crop=raw.crop(region);marked=overlay(raw,anns).crop(region)
            raw_name=f'{split}_image{info["id"]}_ann{ann["id"]}_crop.png'
            crop.save(output/raw_name)
            tiles.extend([(f'{split} image {info["id"]} / annotation {ann["id"]}: raw',crop),
                          (f'{split}: same crop with all GT',marked)])
            source=report['paths'][f'{split}:{info["id"]}']
            review.append(dict(model='dataset_'+split,image_id=info['id'],annotation_id=ann['id'],
                               file_name=info['file_name'],source_group=source['source_group'],
                               leakage_group=source['leakage_group'],bbox_width=ann['bbox'][2],bbox_height=ann['bbox'][3]))
        name=f'{split}_gap_contact_sheet.png';panel(tiles,output/name)
        sections.append(f'<h2>{split}: original source instances, not augmented views</h2><a href="{name}"><img src="{name}"></a>')
    save_review(output/'appearance_review.csv',review)
    html_page(output/'index.html','Dataset review', '<p>Orange = gap GT; other GT colors retained. Compare original training and validation appearances. Box width is not interruption width. Complete appearance_review.csv; no labels or splits are changed.</p>'+''.join(sections))


def gap_panels(output,info,ann,annotations,prepared,search,final_predictions):
    prefix=f'image{info["id"]}_gap{ann["id"]}';raw=prepared['image']
    original=overlay(raw,annotations);region=crop_box(raw,ann)
    original.save(output/f'{prefix}_original_gt.png');raw.crop(region).save(output/f'{prefix}_context_raw.png')
    original.crop(region).save(output/f'{prefix}_context_gt.png')
    prepared['display'].save(output/f'{prefix}_model_input.png')
    input_anns=[]
    for a,box in zip(annotations,prepared['boxes_xyxy']):
        x,y,x2,y2=box;input_anns.append(dict(a,bbox=[x,y,x2-x,y2-y]))
    input_image=overlay(prepared['display'],input_anns);input_image.save(output/f'{prefix}_input_gt.png')
    def candidate_tile(key,title):
        c=search.get(key)
        if not search['available']:return title+' (unknown: raw candidates not captured)',original
        if c is None:return title+' (no qualifying captured candidate)',original
        prediction=dict(c,label=f"ID {c['candidate_id']}: {c['foreground_argmax']} {c['foreground_score']:.3f}; gap {c['gap_score']:.3f}")
        return title+f" | ID {c['candidate_id']}, IoU {c['iou']:.3f}, {c['foreground_argmax']} {c['foreground_score']:.4g}, gap {c['gap_score']:.4g}, background {c['background_score']:.4g}",overlay(raw,annotations,[prediction])
    finals=sorted(final_predictions,key=lambda p:p['score'],reverse=True)
    gt_label=', '.join(f"{a['class_name']} #{a['id']}" for a in annotations)
    tiles=[('Original GT: '+gt_label,original),('Exact model tensor, normalization reversed; GT: '+gt_label,input_image),
           ('Unannotated context crop',raw.crop(region)),('Annotated context crop',original.crop(region)),
           candidate_tile('A','Best overlap, any foreground class'),candidate_tile('B','Best overlap, foreground argmax = gap'),
           candidate_tile('C50','Highest gap score among IoU >= 0.50'),
           (f'Operating final detections: showing {min(8,len(finals))}/{len(finals)}; '+', '.join(f"{p['class_name']} {p['score']:.3f}" for p in finals[:4]),overlay(raw,annotations,finals[:8]))]
    path=f'{prefix}_panel.png';panel(tiles,output/path)
    return dict(panel=path,original_gt=f'{prefix}_original_gt.png',context_raw=f'{prefix}_context_raw.png',
                context_gt=f'{prefix}_context_gt.png',model_input=f'{prefix}_model_input.png',input_gt=f'{prefix}_input_gt.png')


def html_page(path,title,body):
    Path(path).write_text('<!doctype html><meta charset="utf-8"><title>'+html.escape(title)+'</title><style>'
        'body{font:16px/1.5 system-ui;background:#f4f7fb;color:#172b43;margin:30px;max-width:1500px}'
        'img{max-width:100%;height:auto}table{border-collapse:collapse;background:white;width:100%}'
        'td,th{padding:9px;border:1px solid #c5d1e0;vertical-align:top}pre{white-space:pre-wrap;background:white;padding:18px}'
        '.warning{background:#fff1cf;padding:16px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}'
        'a{color:#0953a5}section{background:white;padding:15px;margin:18px 0}</style><h1>'+html.escape(title)+'</h1>'+body,encoding='utf-8')


def model_report(output,manifest,summary,gaps,fp_gallery):
    import json
    comparison=summary.get('saved_prediction_comparison')
    warning=''
    if comparison and not comparison['agrees']:
        warning='<p class="warning">Regenerated predictions disagree with saved baseline predictions. See summary.json. Treat failure observations as provisional; do not attribute them to the saved run without resolving this difference.</p>'
    if manifest['mode']=='saved':warning+='<p class="warning">Saved final predictions only. Full scores, raw candidates and suppression history are unavailable; no pre-filter failure attribution is possible.</p>'
    body=warning+'<p>GT: orange gap, red bridge, blue line collapse, purple multi-bridge. Cyan: prediction. Labels are not visual judgments. IoU 0.25 is a loose diagnostic only. A high-IoU low-confidence candidate is not a successful detection.</p>'
    body+='<p><a href="summary.md">Summary</a> · <a href="manifest.json">Provenance</a> · <a href="manual_review.csv">Editable review CSV</a> · <a href="per_gap.json">Candidate details</a> · <a href="operating_predictions.json">All final predictions with class, score and coordinates</a></p>'
    body+='<pre>'+html.escape(json.dumps({k:summary[k] for k in ('coverage','gap_detection','gap_localization_class_agnostic','all_class_localization','operating_threshold')},indent=2))+'</pre>'
    for g in gaps:
        body+=f'<section><h2>Image {g["image_id"]}, gap annotation {g["annotation_id"]}</h2><a href="{g["visuals"]["panel"]}"><img src="{g["visuals"]["panel"]}"></a>'
        body+='<details><summary>All four candidate searches and measurements</summary><pre>'+html.escape(json.dumps({k:v for k,v in g.items() if k!='visuals'},indent=2))+'</pre></details></section>'
    body+='<h2>False-positive gap detections across the validation subset</h2>'
    if not fp_gallery:body+='<p>No gap false positives at this operating threshold.</p>'
    for item in fp_gallery:body+=f'<p>Image {item["image_id"]}; {item["count"]} gap false positives. Top eight shown.</p><img src="{item["path"]}">'
    html_page(output/'index.html',manifest['experiment_id']+' gap diagnosis',body)


def combined_report(inputs,output,expected,previous=None):
    import csv,json
    reports={read(p/'manifest.json')['experiment_id']:p for p in inputs}
    if len(reports)!=len(inputs):raise ValueError('Duplicate model reports; choose one per model')
    identities={tuple(read(p/'manifest.json')['dataset']['splits'][s]['annotation_sha256'] for s in ('train','val')) for p in inputs}
    if len(identities)>1:raise ValueError('Combined reports must use the same train/validation annotation files')
    body='<p>Validation diagnostics only. Scores across architectures are not calibrated identically. Missing raw candidates and missing models remain explicit.</p>'
    rows=[];all_gaps={};summaries={};markdown=['# Combined validation gap diagnostics','',
        'Read-only workflow. All thresholds are inherited from the baseline metadata. Scores across architectures are not calibrated identically.',
        '', '| Model | Capture | Gap TP/FP/FN | Gap localization | Threshold |', '|---|---|---|---|---|']
    for model in expected:
        if model not in reports:
            body+=f'<p class="warning">{html.escape(model)}: missing/unexecuted</p>';continue
        p=reports[model];s=read(p/'summary.json');summaries[model]=s
        manifest=read(p/'manifest.json');mode=manifest['mode']
        link=os.path.relpath(p/'index.html',output).replace('\\','/')
        counts='/'.join(str(s['gap_detection'][k]) for k in ('tp','fp','fn'))
        loc=f"{s['gap_localization_class_agnostic']['tp']}/{s['gap_localization_class_agnostic']['support']}"
        body+=f'<p><a href="{link}">{html.escape(model)}</a>: gap TP/FP/FN {counts}; threshold {s["operating_threshold"]}; {s["coverage"]}; mode: {mode}</p>'
        markdown.append(f'| {model} | {mode} | {counts} | {loc} | {s["operating_threshold"]} |')
        if mode=='saved':body+='<p class="warning">Final saved predictions only; raw candidates, full class distributions and suppression evidence are unavailable.</p>'
        if s.get('saved_prediction_comparison') and not s['saved_prediction_comparison']['agrees']:
            body+='<p class="warning">Regenerated predictions differ from the saved baseline. Inspect the discrepancy and saved-baseline matching results in the model report before interpreting these candidates.</p>'
        with (p/'manual_review.csv').open(encoding='utf-8-sig',newline='') as f:
            rows.extend(dict(row,raw_candidate_evidence='unknown_not_captured' if mode=='saved' else 'available',
                             report_mode=mode) for row in csv.DictReader(f))
        for g in read(p/'per_gap.json'):all_gaps.setdefault((g['image_id'],g['annotation_id']),{})[model]=(p,g)
    for key,models in all_gaps.items():
        body+=f'<h2>Image {key[0]}, annotation {key[1]}</h2><div class="grid">'
        for model in expected:
            body+='<section><h3>'+html.escape(model)+'</h3>'
            if model in models:
                p,g=models[model];relative=os.path.relpath(p/g['visuals']['panel'],output).replace('\\','/')
                if not g['search']['available']:
                    body+='<p class="warning">A/B/C candidate measurements: UNKNOWN (raw anchors not captured). This is missing evidence, not proof of localization failure.</p>'
                body+=f'<a href="{relative}"><img src="{relative}"></a>'
            else:body+='<p>Missing/unexecuted for this annotation.</p>'
            body+='</section>'
        body+='</div>'
    # Existing manual fields survive; explicit previous combined review takes precedence.
    save_review(output/'manual_review.csv',rows,previous)
    write(output/'summary.json',summaries)
    markdown+=['','All inputs use identical train/validation annotation hashes (checked). Each model manifest records checkpoint identity, dataset paths, transforms, postprocessing and packages.',
        'Prediction discrepancies flagged in the individual summaries make candidate-level attribution to saved baseline results provisional.',
        'Missing instrumentation by model: '+json.dumps({m:s['missing_evidence'] for m,s in summaries.items()}),
        f'Manual review rows: {len(rows)}. Complete manual_review.csv and appearance_review.csv from the dataset report. No definitive root cause assigned.',
        'A candidate may be reused in per-GT searches; final TP counts are one-to-one. Gap-localization counts ignore predicted class and are not class-aware detections.']
    (output/'summary.md').write_text('\n'.join(markdown),encoding='utf-8')
    body='<p><a href="summary.md">Combined summary</a> · <a href="manual_review.csv">Editable combined review</a></p>'+body
    html_page(output/'index.html','Four-model gap diagnostic comparison',body)
