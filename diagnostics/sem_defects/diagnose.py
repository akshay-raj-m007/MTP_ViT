"""Read-only SEM gap diagnostics. No test split or training command is exposed."""
import sys
sys.dont_write_bytecode=True
import argparse
import csv
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
from datetime import datetime
import uuid
from core import ROOT,HERE,OUTPUT,MANUAL,read,write,sha,resolve,new_output,inspect_data,category_mapping,match,gap_search,save_review,compare_predictions


def automatic_observations(search,matched):
    if not search['available']: return ['Only saved final predictions available; pre-filter cause unknown.']
    notes=[];a=search['A'];c=search['C50']
    if a is None:return ['No decoded candidate.']
    notes.append(f"Best geometric overlap IoU={a['iou']:.4f}; foreground={a['foreground_argmax']} score={a['foreground_score']:.4g}; background={a['background_score']:.4g}.")
    if a['background_dominates']:notes.append('Best-overlap candidate is background-dominated; overlap alone is not successful detection.')
    if c is None:notes.append('No decoded candidate reaches IoU 0.50.')
    else:
        notes.append(f"Highest gap score at IoU>=0.50 is {c['gap_score']:.4g}; gap-stage reason={c['stages']['gap']['removal_reason']}.")
    notes.append('Matched by final one-to-one gap detection.' if matched else 'Unmatched by final one-to-one gap detection.')
    return notes


def package_versions():
    result={}
    for name in ('torch','torchvision','transformers','numpy','Pillow','pycocotools'):
        try:result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result[name]=None
    return result


def diagnose(args,config,out,datasets,report):
    import torch
    from adapters import Adapter,model_metadata
    from visuals import gap_panels,model_report,overlay,panel
    torch.set_num_threads(4)
    metadata=model_metadata(config,report)
    adapter=Adapter(config,metadata,args.device,args.mode=='inference')
    capture_source=None
    if args.mode=='captured':
        if not args.capture_report:raise ValueError('--capture-report required for captured mode')
        capture_source=resolve(args.capture_report);prior=read(capture_source/'manifest.json')
        if prior['mode'] not in ('inference','captured') or prior['experiment_id']!=config['experiment_id']:
            raise ValueError('Capture must be from the same experiment with raw candidates')
        identity=lambda files:sorted((Path(f['path']).name,f['sha256']) for f in files)
        if identity(prior['model']['checkpoint_files'])!=identity(metadata['checkpoint_files']):
            raise ValueError('Captured checkpoint identity differs')
        if prior['model']['pipeline_configuration']!=metadata['pipeline_configuration'] or prior['model']['preprocessing_postprocessing']!=metadata['preprocessing_postprocessing']:
            raise ValueError('Captured preprocessing/postprocessing settings differ')
        if any(prior['dataset']['splits'][s]['annotation_sha256']!=report['splits'][s]['annotation_sha256'] for s in ('train','val')):
            raise ValueError('Captured annotation fingerprint differs')
    val=datasets['val']; source_map=category_mapping(val['categories'])
    all_ids={im['id'] for im in val['images']}
    selected=set(args.image_ids) if args.image_ids else all_ids
    if not selected<=all_ids:raise ValueError('Only validation image IDs may be requested')
    if capture_source and not selected<=set(prior['selected_validation_image_ids']):
        raise ValueError('Capture does not contain every requested validation image')
    annotations=[dict(a,class_name=source_map[a['category_id']]) for a in val['annotations'] if a['image_id'] in selected]
    gap_gt=[a for a in annotations if a['class_name']=='gap']
    saved_raw=read(metadata['prediction_path']);saved=[]
    for index,p in enumerate(saved_raw):
        if p['image_id'] not in all_ids:raise ValueError('Saved prediction references image outside validation')
        if p['category_id'] not in metadata['external_labels']:raise ValueError('Unsupported saved prediction category')
        if len(p['bbox'])!=4 or not all(__import__('math').isfinite(v) for v in p['bbox']+[p['score']]):raise ValueError('Nonfinite saved prediction')
        if min(p['bbox'][2:])<0 or not 0<=p['score']<=1:raise ValueError('Invalid saved prediction')
        if p['image_id'] in selected:
            name=metadata['external_labels'][p['category_id']]
            saved.append(dict(p,category_id=adapter.source_ids[name],class_name=name,
                candidate_id=None,detection_id=f'saved:{index}'))
    manifest=dict(experiment_id=config['experiment_id'],mode=args.mode,predictions='regenerated' if args.mode=='inference' else 'loaded',
        device=args.device,python=sys.version,versions=package_versions(),model=metadata,dataset=report,
        resolved_config={k:str(resolve(config[k])) for k in ('train','val','image_root','manifest','run','checkpoint','predictions')},
        selected_validation_image_ids=sorted(selected),script_hashes={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*.py')},
        readonly=True,inference_mode='eval + torch.inference_mode; fp32',created=datetime.now().isoformat())
    if capture_source:
        manifest['capture_source']=str(capture_source);manifest['capture_source_manifest_sha256']=sha(capture_source/'manifest.json')
        manifest['capture_inference_device']=prior.get('capture_inference_device',prior['device'])
        manifest['capture_inference_versions']=prior.get('capture_inference_versions',prior['versions'])
    write(out/'manifest.json',manifest)
    images=[im for im in val['images'] if im['id'] in selected]
    prepared_all={};raw_by_image={};baseline=[];capture_reports={};roundtrips={}
    for info in images:
        anns=[a for a in annotations if a['image_id']==info['id']]
        prepared=adapter.preprocess(info,anns,resolve(config['image_root']));prepared_all[info['id']]=prepared
        roundtrips[str(info['id'])]=prepared['roundtrip_max_error_pixels']
        if args.mode=='inference':
            rows,pred,cap=adapter.capture(info,prepared);baseline.extend(pred);capture_reports[str(info['id'])]=cap
            raw_by_image[info['id']]=rows
            with gzip.open(out/f'candidates_image{info["id"]}.json.gz','wt',encoding='utf-8') as f:
                json.dump(dict(image_id=info['id'],metadata=cap,candidates=rows),f,allow_nan=False)
        elif capture_source:
            import shutil
            filename=f'candidates_image{info["id"]}.json.gz'
            with gzip.open(capture_source/filename,'rt',encoding='utf-8') as f:captured=json.load(f)
            raw_by_image[info['id']]=captured['candidates'];capture_reports[str(info['id'])]=captured['metadata']
            shutil.copyfile(capture_source/filename,out/filename)
        else: raw_by_image[info['id']]=None
        print(f"{config['experiment_id']}: image {info['id']} processed ({args.mode})",flush=True)
    if args.mode=='saved':baseline=saved
    if capture_source:baseline=[p for p in read(capture_source/'baseline_predictions.json') if p['image_id'] in selected]
    threshold=metadata['preprocessing_postprocessing']['operating_threshold']
    finals=[p for p in baseline if p['score']>=threshold]
    gaps_final=[p for p in finals if p['class_name']=='gap']
    gap_metrics=match(gap_gt,gaps_final)
    localization=match(gap_gt,finals,False)  # FPs here mean unmatched to GAP, not all-defect FPs.
    all_localization=match(annotations,finals,False)
    comparison=compare_predictions(baseline,saved,selected) if args.mode!='saved' else None
    saved_finals=[p for p in saved if p['score']>=threshold]
    saved_gap_metrics=match(gap_gt,[p for p in saved_finals if p['class_name']=='gap'])
    saved_localization=match(gap_gt,saved_finals,False)
    positional_comparison=None
    if args.mode!='saved' and config['family']=='yolos':
        # The unchanged YOLOS exporter writes queries in order, but does not store IDs.
        pairs=[];ordered=True
        for image_id in selected:
            a=[p for p in baseline if p['image_id']==image_id];b=[p for p in saved if p['image_id']==image_id]
            if len(a)!=len(b):ordered=False;break
            pairs.extend(zip(a,b))
        if ordered and pairs:
            positional_comparison=dict(assumption='saved YOLOS query order retained by existing decoder/exporter; saved IDs unavailable',
                max_box_delta_pixels=max(abs(x-y) for a,b in pairs for x,y in zip(a['bbox'],b['bbox'])),
                max_score_delta=max(abs(a['score']-b['score']) for a,b in pairs),
                foreground_class_disagreements=sum(a['category_id']!=b['category_id'] for a,b in pairs))
    summary=dict(coverage='full_validation' if selected==all_ids else 'limited_smoke_subset_not_full_validation',
        validation_images_analyzed=len(images),validation_images_total=len(all_ids),gap_detection=gap_metrics,
        gap_localization_class_agnostic=localization,all_class_localization=all_localization,
        operating_threshold=threshold,matching='descending score, IoU>=0.50, each prediction and GT at most once; all selected validation images contribute FPs',
        gap_localization_warning='FP in gap-only class-agnostic matching means unmatched to a gap, not necessarily a false defect detection.',
        saved_prediction_comparison=comparison,roundtrip_max_error_pixels=max(roundtrips.values(),default=0),
        positional_saved_comparison=positional_comparison,saved_baseline_gap_detection=saved_gap_metrics,
        saved_baseline_gap_localization=saved_localization,
        roundtrips=roundtrips,capture_reports=capture_reports,
        missing_evidence=[] if args.mode!='saved' else ['pre-filter candidates','full class scores','background scores','query/anchor IDs','filter/NMS lineage'],
        manual_review_pending=len(gap_gt))
    matched={m['annotation_id'] for m in gap_metrics['matches']};gaps=[];review=[];fp_gallery=[]
    for info in images:
        anns=[a for a in annotations if a['image_id']==info['id']];prepared=prepared_all[info['id']]
        for idx,ann in enumerate(anns):
            if ann['class_name']!='gap':continue
            search=gap_search(ann,raw_by_image[info['id']]);box=prepared['boxes_xyxy'][idx];x,y,w,h=ann['bbox']
            measurements=dict(original_width=info['width'],original_height=info['height'],input_width=prepared['tensor'].shape[2],
                 input_height=prepared['tensor'].shape[1],bbox_width=w,bbox_height=h,input_bbox_width=box[2]-box[0],input_bbox_height=box[3]-box[1],
                 border_distances_pixels=dict(left=x,top=y,right=info['width']-x-w,bottom=info['height']-y-h),
                 transform=prepared['meta'],roundtrip_max_error_pixels=prepared['roundtrip_max_error_pixels'])
            visuals=gap_panels(out,info,ann,anns,prepared,search,[p for p in finals if p['image_id']==info['id']])
            observations=automatic_observations(search,ann['id'] in matched)
            gaps.append(dict(image_id=info['id'],annotation_id=ann['id'],measurements=measurements,search=search,
                             independent_searches_can_reuse_candidates=True,automatic_observations=observations,visuals=visuals))
            row=dict(model=config['experiment_id'],image_id=info['id'],annotation_id=ann['id'],file_name=info['file_name'],
                     original_bbox_width=w,original_bbox_height=h,input_bbox_width=box[2]-box[0],input_bbox_height=box[3]-box[1],
                     roundtrip_error=prepared['roundtrip_max_error_pixels'],threshold=threshold,
                     candidate_capture_available=search['available'],final_gap_matched=ann['id'] in matched,
                     automated_observations=' | '.join(observations))
            for key in ('A','B','C50','C25'):
                c=search[key]
                for field in ('candidate_id','iou','foreground_argmax','foreground_score','gap_score','background_score','background_dominates','gap_final'):
                    row[f'{key}_{field}']=c.get(field) if c else None
            review.append(row)
        fps=[p for p in gap_metrics['false_positives'] if p['image_id']==info['id']]
        if fps:
            name=f'false_gap_image{info["id"]}.png'
            panel([(f'Image {info["id"]}: top {min(8,len(fps))}/{len(fps)} false gap detections',overlay(prepared['image'],anns,sorted(fps,key=lambda p:p['score'],reverse=True)[:8]))],out/name)
            fp_gallery.append(dict(image_id=info['id'],count=len(fps),path=name))
    write(out/'baseline_predictions.json',baseline);write(out/'operating_predictions.json',finals)
    write(out/'summary.json',summary);write(out/'per_gap.json',gaps)
    save_review(out/'manual_review.csv',review,args.previous_review)
    lines=[f"# {config['experiment_id']} validation gap diagnosis",'',f"Mode: {args.mode}; coverage: {summary['coverage']}; device: {args.device}; precision: fp32.",
        f"Validation images: {len(images)}/{len(all_ids)}; gap instances: {len(gap_gt)}. Operating threshold: {threshold}.",
        f"Class-aware gap TP/FP/FN: {gap_metrics['tp']}/{gap_metrics['fp']}/{gap_metrics['fn']}; precision: {gap_metrics['precision']}; recall: {gap_metrics['recall']}.",
        f"Class-agnostic gap localization: {localization['tp']}/{len(gap_gt)}; all-class localization: {all_localization['tp']}/{len(annotations)}.",
        f"Coordinate round-trip maximum error: {summary['roundtrip_max_error_pixels']:.8f} pixels.",
        f"Dataset check passes: {', '.join(report['checks'])}. Exclusions: none.",
        f"Checkpoint: {metadata['checkpoint']}; available: {metadata['checkpoint_available']}. Full hashes/configuration in manifest.json.",
        f"Reviewed manifest fingerprint: {report['declared_split_fingerprint']}.",
        f"Saved prediction comparison: {json.dumps(comparison) if comparison is not None else 'Not regenerated; agreement untested.'}",
        f"YOLOS ordered comparison (explicit assumption): {json.dumps(positional_comparison)}.",
        f"Saved-baseline gap TP/FP/FN: {saved_gap_metrics['tp']}/{saved_gap_metrics['fp']}/{saved_gap_metrics['fn']}; saved class-agnostic gap matches: {saved_localization['tp']}/{len(gap_gt)}.",
        f"Missing evidence: {', '.join(summary['missing_evidence']) or 'none for captured model stages'}.",
        '', 'Independent best-candidate searches may reuse one box; final metrics are one-to-one. IoU 0.25 is not a correct detection criterion.',
        'Full candidate dumps are stored once per image. Confidence scores are not calibrated across architectures.',
        'No visual root cause is assigned automatically. Complete manual_review.csv; box width is not visible interruption width.']
    if comparison and not comparison['agrees']:lines+=['','**Prediction discrepancy: do not attribute these regenerated observations to the saved baseline until resolved.**']
    for g in gaps:lines+=['',f"## Image {g['image_id']}, annotation {g['annotation_id']}"]+['- '+n for n in g['automatic_observations']]
    lines+=['',f'Manual review pending: {len(gaps)} rows. Review visibility, ambiguity, adjacent instances, and training/validation appearance.']
    (out/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    model_report(out,manifest,summary,gaps,fp_gallery)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['inspect','run','combine','review-summary'])
    p.add_argument('--config',type=Path,default=HERE/'configs/yolos_unweighted.json')
    p.add_argument('--output',type=Path)
    p.add_argument('--mode',choices=['inference','saved','captured'],default='inference')
    p.add_argument('--capture-report',type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--image-ids',type=int,nargs='+')
    for key in ('checkpoint','train','val','image-root','manifest','predictions','run'):
        p.add_argument('--'+key,dest=key.replace('-','_'))
    p.add_argument('--experiment-id');p.add_argument('--family',choices=['yolos','ssd'])
    p.add_argument('--reports',type=Path,nargs='+');p.add_argument('--previous-review',type=Path)
    p.add_argument('--review',type=Path)
    args=p.parse_args()
    if args.command=='review-summary':
        if not args.review:p.error('--review is required')
        with args.review.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        from collections import Counter
        reviewed=[r for r in rows if r.get('reviewed','').strip().lower() in ('yes','true','1')]
        allowed=dict(annotation_correct={'yes','no','uncertain'},original_visibility={'clear','subtle','unclear'},
                     input_visibility={'clear','subtle','lost','uncertain'},primary_failure={'annotation_ambiguity','resolution_or_visibility_limitation',
                     'localization_failure','localization_imprecision','class_confusion','low_confidence','postprocessing_removal',
                     'instance_merging','successful_detection','uncertain'})
        incomplete=[dict(model=r['model'],image_id=r['image_id'],annotation_id=r['annotation_id'],
                        invalid_or_missing=[k for k,choices in allowed.items() if r.get(k,'') not in choices])
                    for r in reviewed if any(r.get(k,'') not in choices for k,choices in allowed.items())]
        result=dict(rows=len(rows),reviewed=len(reviewed),pending=len(rows)-len(reviewed),
            human_primary_failures=dict(Counter(r.get('primary_failure') or 'unspecified' for r in reviewed)),
            marked_reviewed_but_incomplete=incomplete,
            note='Human judgments only; source CSV unchanged.')
        print(json.dumps(result,indent=2));return
    config=read(args.config)
    for key in ('checkpoint','train','val','image_root','manifest','predictions','run','experiment_id','family'):
        if getattr(args,key,None) is not None:config[key]=getattr(args,key)
    out=new_output(args.output or OUTPUT/(args.command+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]))
    try:
        if args.command=='combine':
            if not args.reports:p.error('--reports is required')
            from visuals import combined_report
            combined_report([resolve(x) for x in args.reports],out,['yolos_unweighted','yolos_weighted','ssd_unweighted','ssd_weighted'],args.previous_review)
        else:
            datasets,report=inspect_data(resolve(config['train']),resolve(config['val']),resolve(config['image_root']),resolve(config['manifest']))
            write(out/'dataset_checks.json',report)
            if args.command=='inspect':
                from visuals import dataset_gallery
                dataset_gallery(datasets,resolve(config['image_root']),out,report)
            else:diagnose(args,config,out,datasets,report)
    except Exception as exc:
        write(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),status='failed; no interpretation available'))
        raise
    print(f'Report: {out / "index.html"}',flush=True)


if __name__=='__main__':main()
