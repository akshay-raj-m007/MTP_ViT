"""Validate COCO and prepare reproducible group-safe splits without changing raw data."""

# Allow direct execution without installing the project package.
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import json
import hashlib
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
from sem_yolos.common import ROOT, CLASSES, CATEGORIES, read, write, resolve_image, raw_images_dir, prepared_fingerprint

CONFIRMED_CLEAN = "c5329556-fig34.jpg"

def mapping(categories):
    if len({c["id"] for c in categories}) != len(categories):
        raise ValueError("Duplicate category IDs")
    if len({c["name"] for c in categories}) != len(categories):
        raise ValueError("Duplicate category names")
    return {c["id"]: CLASSES.index(c["name"]) for c in categories if c["name"] in CLASSES}

def inspect(root, source):
    d = read(source)
    if not isinstance(d, dict) or not all(k in d for k in ("images", "categories", "annotations")):
        raise ValueError("This input is not COCO. Export COCO or implement/review native conversion before proceeding.")
    errors = []
    for key in ("images", "annotations"):
        if len({x["id"] for x in d[key]}) != len(d[key]): errors.append(f"Duplicate {key} IDs")
    remap = mapping(d["categories"])
    cats = {c["id"]: c["name"] for c in d["categories"]}
    ids = {i["id"] for i in d["images"]}
    by_image = defaultdict(list)
    boxes = []
    for a in d["annotations"]:
        if a["image_id"] not in ids: errors.append(f"Orphan annotation {a['id']}")
        if a["category_id"] not in remap:
            errors.append(f"Annotation {a['id']}: unsupported box label {cats.get(a['category_id'])}; no_defect cannot label a box")
            continue
        if a.get("ignore",0) or a.get("iscrowd",0): errors.append(f"Annotation {a['id']}: crowd/ignore unsupported; resolve explicitly")
        b = dict(a, category_id=remap[a["category_id"]], iscrowd=0)
        by_image[a["image_id"]].append(b)
        boxes.append(b)
    records, hashes, sizes, signatures = [], defaultdict(list), Counter(), {}
    for im in d["images"]:
        try: path = resolve_image(root, im["file_name"])
        except ValueError as e:
            errors.append(str(e)); continue
        with Image.open(path) as raw:
            pic = raw.convert("RGB"); w,h = pic.size
            digest = hashlib.sha256(f"{w},{h}:".encode()+pic.tobytes()).hexdigest()
            pixels = list(pic.convert("L").resize((9,8)).get_flattened_data())
            signatures[im["id"]] = sum(int(pixels[y*9+x]>pixels[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))
        sizes[f"{w}x{h}"] += 1
        if (w,h) != (im["width"],im["height"]): errors.append(f"Image {im['id']}: dimensions disagree")
        for a in by_image[im["id"]]:
            b = a["bbox"]
            if len(b)!=4 or not all(isinstance(v,(float,int)) and math.isfinite(v) for v in b):
                errors.append(f"Annotation {a['id']}: malformed/nonfinite box"); continue
            x,y,bw,bh = b
            if min(x,y)<0 or min(bw,bh)<=0 or x+bw>w+1e-6 or y+bh>h+1e-6:
                errors.append(f"Annotation {a['id']}: invalid/out-of-bounds box {b}")
            a["area"] = bw*bh
        labels = sorted({a["category_id"] for a in by_image[im["id"]]})
        confirmed_clean = path.name == CONFIRMED_CLEAN and not labels
        record = dict(im, file_name=path.relative_to(root).as_posix(), classes=labels, sha256=digest,
                      file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      annotation_status="user_confirmed_clean" if confirmed_clean else ("box_annotated" if labels else "assumed_clean_coco"),
                      evaluation_eligible=True)
        records.append(record); hashes[digest].append(im["id"])
    near=[]; keys=list(signatures)
    for pos,a in enumerate(keys):
        for b in keys[pos+1:]:
            distance=(signatures[a]^signatures[b]).bit_count()
            if distance<=5: near.append(dict(ids=[a,b],dhash_distance=distance))
    used={x["file_name"] for x in records}
    extra=[p.relative_to(root).as_posix() for p in raw_images_dir(root).rglob("*") if p.is_file() and p.relative_to(root).as_posix() not in used]
    if extra: errors.append(f"Files absent from export: {extra}")
    report=dict(schema="COCO",images=len(d["images"]),boxes=len(boxes),errors=errors,
                dimensions=dict(sizes),instances=dict(Counter(CLASSES[a["category_id"]] for a in boxes)),
                class_presence=dict(Counter(CLASSES[c] for im in records for c in im["classes"])),
                empty_images=[im["file_name"] for im in records if not im["classes"]],
                exact_duplicate_groups=[v for v in hashes.values() if len(v)>1],near_duplicate_candidates=near,
                source_to_model=remap,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                assumptions=["35 empty images assumed clean from COCO and supplied dataset description; review metadata unavailable.",
                             "fig34 is clean (no_defect) by explicit user confirmation; 36 clean images total, with zero detection boxes.",
                             "Near-duplicate candidates are review hints, not inferred source groups."])
    report['unique_image_ids']=len(ids)==len(d['images'])
    report['unique_annotation_ids']=len({a['id'] for a in d['annotations']})==len(d['annotations'])
    report['crowd_annotations']=sum(bool(a.get('iscrowd',0)) for a in d['annotations'])
    report['ignored_annotations']=sum(bool(a.get('ignore',0)) for a in d['annotations'])
    report['source_file_count']=len(used)+len(extra)
    dimensions={im['id']:(im['width'],im['height']) for im in records}
    scaled_sides=[min(a['bbox'][2:])*min(512/dimensions[a['image_id']][0],512/dimensions[a['image_id']][1]) for a in boxes if a['image_id'] in dimensions]
    report['boxes_shorter_than_4_pixels_at_512']=sum(v<4 for v in scaled_sides)
    report['minimum_box_short_side_at_512']=min(scaled_sides,default=None)
    return records, boxes, report

def split_records(records, seed, source_groups):
    # Union exact duplicate content and explicitly supplied source relationships.
    parent={im["id"]:im["id"] for im in records}
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(a,b): parent[find(b)]=find(a)
    seen={}
    for im in records:
        group=source_groups.get(im["file_name"],source_groups.get(Path(im["file_name"]).name))
        im["source_group"]=group
        for key in [("hash",im["sha256"])]+([("source",group)] if group is not None else []):
            if key in seen: union(im["id"],seen[key])
            else: seen[key]=im["id"]
    groups=defaultdict(list)
    for im in records: groups[find(im["id"])].append(im)
    rng=random.Random(seed)
    assignments={}
    targets={0:[14,4,4],1:[3,2,2],2:[14,3,3],3:[6,2,2],4:[25,5,5]}
    totals=Counter(c for im in records if im["evaluation_eligible"] for c in (im["classes"] or [4]))
    if totals[4] in (35,36): targets[4]=[totals[4]-10,5,5]
    for c,n in totals.items():
        if sum(targets[c])!=n:
            v=max(1,round(n*.17)) if n>=3 else 0
            targets[c]=[n-2*v,v,v]
    achieved=[Counter() for _ in range(3)]
    items=list(groups.values()); rng.shuffle(items)
    items.sort(key=lambda g: (-len(g), min(totals.get(c,1) for im in g for c in (im["classes"] or [4]))))
    simple=all(len(g)==1 and len(g[0]["classes"])<=1 for g in items)
    if simple:
        for c in range(5):
            pool=[g[0] for g in items if g[0]["evaluation_eligible"] and (g[0]["classes"] or [4])==[c]]
            start=0
            for s,n in enumerate(targets[c]):
                for im in pool[start:start+n]: assignments[im["id"]]=s
                start+=n
        for im in records:
            if not im["evaluation_eligible"]: assignments[im["id"]]=0
    else:
        for g in items:
            counts=Counter(c for im in g for c in (im["classes"] or [4]))
            choices=[0] if any(not im["evaluation_eligible"] for im in g) else list(range(3))
            def cost(s):
                return sum(((achieved[k][c]+(counts[c] if k==s else 0)-targets[c][k])/max(1,targets[c][k]))**2 for k in range(3) for c in range(5))
            s=min(choices,key=cost)
            achieved[s].update(counts)
            for im in g: assignments[im["id"]]=s
    for im in records:
        im["split"]=["train","val","test"][assignments[im["id"]]]
        im["leakage_group"]=str(find(im["id"]))
    return records

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--source',default='data/raw/result.json'); p.add_argument('--output',default='data/prepared/label_studio')
    p.add_argument('--seed',type=int,default=42); p.add_argument('--groups',type=Path); p.add_argument('--resplit',action='store_true')
    args=p.parse_args(); root=args.root.resolve(); out=root/args.output
    records,boxes,report=inspect(root,root/args.source)
    # COCOeval uses annotation ID 0 as its unmatched sentinel. Preserve raw IDs separately.
    for index,box in enumerate(boxes,1):
        box['source_annotation_id']=box['id']
        box['id']=index
    write(out/'validation_report.json',report)
    if report['errors']: raise ValueError(report['errors'])
    source_groups=read(args.groups) if args.groups else {}
    unknown=set(source_groups)-{v for im in records for v in (im['file_name'],Path(im['file_name']).name)}
    if unknown: raise ValueError(f'Group entries not found: {unknown}')
    fingerprint=hashlib.sha256(json.dumps([report['source_sha256'],[(i['id'],i['file_sha256']) for i in records],source_groups],sort_keys=True).encode()).hexdigest()
    manifest_path=out/'split_manifest.json'
    if manifest_path.exists() and not args.resplit:
        saved=read(manifest_path)
        if saved['fingerprint']!=fingerprint: raise ValueError('Dataset/groups changed; inspect changes and explicitly use --resplit')
        previous={im['id']:im for im in saved['images']}
        for im in records:
            for key in ('split','source_group','leakage_group'):
                im[key]=previous[im['id']][key]
        write(manifest_path,dict(seed=saved['seed'],fingerprint=fingerprint,images=records))
    else:
        records=split_records(records,args.seed,source_groups)
        write(manifest_path,dict(seed=args.seed,fingerprint=fingerprint,images=records))
    distributions={}; box_distributions={}
    for split in ['train','val','test']:
        subset=[im for im in records if im['split']==split]; ids={im['id'] for im in subset}
        write(out/(split+'.json'),dict(info={'source':'Label Studio'},images=subset,annotations=[a for a in boxes if a['image_id'] in ids],categories=CATEGORIES))
        distributions[split]=dict(Counter('deliberately_unannotated' if not im['evaluation_eligible'] else (CLASSES[c] if c<4 else 'no_defect') for im in subset for c in (im['classes'] or [4])))
        box_distributions[split]=dict(Counter(CLASSES[a['category_id']] for a in boxes if a['image_id'] in ids))
    manifest=read(manifest_path)
    manifest['split_fingerprint']=prepared_fingerprint(out)
    write(manifest_path,manifest)
    write(out/'class_mapping.json',dict(id2label=dict(enumerate(CLASSES)),source_to_model=report['source_to_model']))
    report['split_distributions']=distributions
    report['split_box_distributions']=box_distributions
    lookup={im['id']:im for im in records}
    report['near_duplicate_cross_split_candidates']=sum(lookup[pair['ids'][0]]['split']!=lookup[pair['ids'][1]]['split'] for pair in report['near_duplicate_candidates'])
    lines=['# Near-duplicate review candidates','','dHash distance <= 5 is screening, not proof of shared source. No groups inferred.','', '| Image A | Split | Image B | Split | Distance |','|---|---|---|---|---|']
    for pair in report['near_duplicate_candidates']:
        a,b=[lookup[i] for i in pair['ids']]
        lines.append(f"| {a['file_name']} | {a['split']} | {b['file_name']} | {b['split']} | {pair['dhash_distance']} |")
    (out/'NEAR_DUPLICATE_REVIEW.md').write_text('\n'.join(lines),encoding='utf-8')
    write(out/'validation_report.json',report)
    text='# Dataset report\n\n```json\n'+json.dumps({k:report[k] for k in ['images','boxes','instances','class_presence','split_distributions','split_box_distributions']},indent=2)+'\n```\n\n'
    text+='\n\n'.join(report['assumptions'])
    text+='\n\nSource groups supplied: '+str(bool(source_groups))
    text+='\n\nFull dimensions, hashes and near-duplicate candidates: data/prepared/label_studio/validation_report.json and split_manifest.json. Small originals include 81x81 pixels; upscaling cannot recover detail. No test predictions were computed.\n'
    text+='\n\nValidation: '+json.dumps({k:report[k] for k in ['errors','unique_image_ids','unique_annotation_ids','crowd_annotations','ignored_annotations','exact_duplicate_groups','near_duplicate_cross_split_candidates','boxes_shorter_than_4_pixels_at_512','minimum_box_short_side_at_512']},indent=2)
    text+='\n\nNear-duplicate candidates: '+str(len(report['near_duplicate_candidates']))+'. See NEAR_DUPLICATE_REVIEW.md before interpreting source-independent generalization.\n'
    (root/'docs').mkdir(parents=True,exist_ok=True)
    (root/'docs/DATASET_REPORT.md').write_text(text,encoding='utf-8')
    print(json.dumps(distributions,indent=2))

if __name__=='__main__': main()
