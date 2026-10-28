"""Read-only data checks, matching and candidate searches. No training imports."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT/'outputs/diagnostics_sem'
MANUAL = ['annotation_correct','original_visibility','input_visibility','primary_failure',
          'secondary_failure','reviewer_notes','reviewed','single_or_adjacent_gaps',
          'gap_size_relative_to_strip_width','pitch_orientation_differences',
          'contrast_noise_differences','annotation_consistency','missing_training_appearances']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def resolve(path):
    return (ROOT/Path(path)).resolve()


def new_output(path):
    path=resolve(path)
    if not path.is_relative_to(OUTPUT.resolve()):
        raise ValueError('Output must be under outputs/diagnostics_sem/')
    path.mkdir(parents=True,exist_ok=False)
    return path


def category_mapping(categories):
    ids=[c['id'] for c in categories]; names=[c['name'] for c in categories]
    if len(set(ids))!=len(ids) or len(set(names))!=len(names): raise ValueError('Duplicate category IDs/names')
    mapping={c['id']:c['name'] for c in categories if c['name']!='no_defect'}
    if set(mapping.values()) != {'bridge','gap','line_collapse','multi_bridge'}:
        raise ValueError('Expected four named defect classes; no_defect must have no boxes')
    return mapping


def image_path(root, info):
    root=Path(root).resolve()
    candidates=[(root/info['file_name']).resolve(),(root/Path(info['file_name']).name).resolve()]
    valid=list(dict.fromkeys(p for p in candidates if p.is_file() and p.is_relative_to(root)))
    if len(valid)!=1: raise ValueError(f'Missing/ambiguous image path: {info["file_name"]}')
    return valid[0]


def inspect_data(train_path, val_path, image_root, manifest_path):
    from PIL import Image
    report=dict(checks=[],exclusions=[],splits={},paths={},errors=[])
    datasets={}; seen_ids=set(); seen_files={}; seen_groups={}
    manifest=read(manifest_path) if Path(manifest_path).is_file() else {}
    metadata={im['id']:im for im in manifest.get('images',[]) if im.get('split') in ('train','val')}
    mappings=[]
    for split,path in [('train',train_path),('val',val_path)]:
        path=Path(path).resolve()
        if 'test' in path.stem.lower(): raise ValueError('Test data is forbidden in this diagnostic workflow')
        data=read(path); datasets[split]=data
        mapping=category_mapping(data['categories']); mappings.append(mapping)
        images={im['id']:im for im in data['images']}
        if len(images)!=len(data['images']) or set(images)&seen_ids: raise ValueError('Duplicate/overlapping image IDs')
        seen_ids.update(images); ann_ids=set(); counts={i:0 for i in images}
        for ann in data['annotations']:
            if ann['id'] in ann_ids: raise ValueError('Duplicate annotation IDs')
            ann_ids.add(ann['id'])
            if ann['image_id'] not in images or ann['category_id'] not in mapping:
                raise ValueError(f'Orphan/unsupported/no_defect annotation: {ann}')
            if ann.get('iscrowd',0) or ann.get('ignore',0): raise ValueError('Crowd/ignored GT unsupported; nothing silently dropped')
            box=ann['bbox']; im=images[ann['image_id']]
            if len(box)!=4 or not all(math.isfinite(x) for x in box): raise ValueError(f'Invalid bbox {ann}')
            x,y,w,h=box
            if min(w,h)<=0 or min(x,y)<0 or x+w>im['width']+1e-6 or y+h>im['height']+1e-6:
                raise ValueError(f'Nonpositive/out-of-bounds annotation: {ann}')
            counts[ann['image_id']]+=1
        for image_id,info in images.items():
            if info.get('split',split)!=split or not info.get('evaluation_eligible',True):
                raise ValueError('Wrong split/ineligible image; no automatic exclusions')
            path_im=image_path(image_root,info)
            with Image.open(path_im) as im:
                if im.size!=(info['width'],info['height']): raise ValueError(f'Dimension mismatch: {path_im}')
                im.load()
            h=sha(path_im)
            if h in seen_files and seen_files[h]!=split: raise ValueError('Image contents overlap splits')
            seen_files[h]=split
            source=metadata.get(image_id,{})
            if source and source['split']!=split: raise ValueError('Manifest membership mismatch')
            if source.get('file_sha256',h)!=h: raise ValueError('Image hash mismatch against reviewed manifest')
            for field in ('source_group','leakage_group'):
                if source.get(field) is not None:
                    key=(field,str(source[field]))
                    if key in seen_groups and seen_groups[key]!=split: raise ValueError('Source group crosses train/val')
                    seen_groups[key]=split
            report['paths'][f'{split}:{image_id}']=dict(path=str(path_im),sha256=h,
                source_group=source.get('source_group'),leakage_group=source.get('leakage_group'))
        gap=[a for a in data['annotations'] if mapping[a['category_id']]=='gap']
        report['splits'][split]=dict(images=len(images),boxes=len(data['annotations']),empty_images=sum(v==0 for v in counts.values()),
             gap_images=len({a['image_id'] for a in gap}),gap_instances=len(gap),annotation_sha256=sha(path),annotation_path=str(path))
    if mappings[0]!=mappings[1]: raise ValueError('Train/validation category mappings differ')
    report.update(categories=mappings[0],manifest_path=str(Path(manifest_path).resolve()),
                  manifest_sha256=sha(manifest_path) if manifest else None,
                  declared_split_fingerprint=manifest.get('split_fingerprint'),
                  fingerprint_note='Train/val bytes hashed here. Historical manifest fingerprint recorded; test JSON not opened or recomputed.',
                  checks=['references','unique IDs','dimensions','finite positive in-bounds boxes','image hashes',
                          'train/val categories','empty targets','available group separation'])
    return datasets, report


def iou(a,b):
    x,y,w,h=a; X,Y,W,H=b
    intersection=max(0,min(x+w,X+W)-max(x,X))*max(0,min(y+h,Y+H)-max(y,Y))
    union=w*h+W*H-intersection
    return intersection/union if union>0 else 0.


def match(annotations,predictions,class_aware=True):
    """Score-ordered greedy one-to-one IoU >= .50. Predictions include all val images."""
    if len({p['detection_id'] for p in predictions})!=len(predictions):
        raise ValueError('Duplicate detection IDs; a prediction cannot be counted twice')
    used=set(); matches=[]; false=[]
    for p in sorted(predictions,key=lambda x:x['score'],reverse=True):
        options=[(iou(p['bbox'],a['bbox']),j) for j,a in enumerate(annotations)
                 if j not in used and p['image_id']==a['image_id'] and
                 (not class_aware or p['class_name']==a['class_name'])]
        overlap,index=max(options,default=(0.,-1))
        if overlap>=.5:
            used.add(index); matches.append(dict(prediction_id=p['detection_id'],annotation_id=annotations[index]['id'],
                                                image_id=p['image_id'],iou=overlap))
        else: false.append(p)
    tp=len(matches);fp=len(false);fn=len(annotations)-tp
    return dict(tp=tp,fp=fp,fn=fn,support=len(annotations),predictions=len(predictions),
                precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else None,
                matches=matches,false_positives=false,
                unmatched_annotation_ids=[a['id'] for i,a in enumerate(annotations) if i not in used])


def gap_search(annotation,candidates):
    """Independent per-GT searches; deliberately may reuse a candidate across GTs."""
    if candidates is None: return dict(available=False,A=None,B=None,C50=None,C25=None)
    def result(c):
        return dict(c, iou=iou(annotation['bbox'],c['bbox'])) if c is not None else None
    relevant=[c for c in candidates if c['foreground_argmax']=='gap']
    at50=[c for c in candidates if iou(annotation['bbox'],c['bbox'])>=.5]
    at25=[c for c in candidates if iou(annotation['bbox'],c['bbox'])>=.25]
    return dict(available=True,
        A=result(max(candidates,key=lambda c:iou(annotation['bbox'],c['bbox']),default=None)),
        B=result(max(relevant,key=lambda c:iou(annotation['bbox'],c['bbox']),default=None)),
        C50=result(max(at50,key=lambda c:c['gap_score'],default=None)),
        C25=result(max(at25,key=lambda c:c['gap_score'],default=None)))


def save_review(path, rows, previous=None):
    """Create-only, preserve manual fields from a previous explicitly supplied review."""
    old={}
    if previous:
        with Path(previous).open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                key=(row['model'],row['image_id'],row['annotation_id'])
                if key in old: raise ValueError('Duplicate manual-review keys')
                old[key]=row
    fields=list(dict.fromkeys(k for row in rows for k in row))+MANUAL
    fields=list(dict.fromkeys(fields))
    with Path(path).open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            key=(str(row['model']),str(row['image_id']),str(row['annotation_id']))
            manual={k:old.get(key,{}).get(k,row.get(k,'')) for k in MANUAL}
            writer.writerow(dict(row,**manual))


def compare_predictions(actual, saved, image_ids, tolerance_box=1e-3,tolerance_score=1e-5):
    """Unordered matching with strict count, label, box and score tolerance."""
    saved=[p for p in saved if p['image_id'] in image_ids]
    used=set(); errors=[]; max_box=max_score=0.
    for p in actual:
        options=[]
        for j,q in enumerate(saved):
            if j not in used and p['image_id']==q['image_id'] and p['category_id']==q['category_id']:
                box=max(abs(a-b) for a,b in zip(p['bbox'],q['bbox']));score=abs(p['score']-q['score'])
                if box<=tolerance_box and score<=tolerance_score: options.append((box+score,j,box,score))
        if options:
            _,j,box,score=min(options);used.add(j);max_box=max(max_box,box);max_score=max(max_score,score)
        else: errors.append({k:p[k] for k in ('image_id','category_id','bbox','score')})
    return dict(agrees=not errors and len(used)==len(saved),regenerated=len(actual),saved=len(saved),
                unmatched_regenerated=len(errors),unmatched_saved=len(saved)-len(used),
                box_tolerance_pixels=tolerance_box,score_tolerance=tolerance_score,
                max_matched_box_error=max_box,max_matched_score_error=max_score,
                discrepancy_examples=errors[:5])
