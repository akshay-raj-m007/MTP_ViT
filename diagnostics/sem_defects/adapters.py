"""Diagnostic adapters around existing, unchanged baseline preprocessing/decoding."""
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from core import ROOT, read, sha, image_path, resolve

sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'experiments/mobilevitv2_ssd'))


def model_metadata(config, dataset_report):
    run=resolve(config['run']); family=config['family']; checkpoint=resolve(config['checkpoint'])
    mapping={int(k):v for k,v in dataset_report['categories'].items()}
    if family=='yolos':
        cfg=read(checkpoint/'pipeline_config.json'); mc=read(checkpoint/'config.json')
        labels={int(k):v for k,v in mc['id2label'].items()}
        threshold_info=read(checkpoint/'threshold.json'); threshold=threshold_info['threshold']
        identity=read(checkpoint/'data_identity.json')['fingerprint']
        source=read(run/'validation/provenance.json')
        if source['split']!='val': raise ValueError('Saved predictions are not validation')
        if source['data_fingerprint']!=identity: raise ValueError('YOLOS prediction fingerprint mismatch')
        files=list(checkpoint.glob('*.safetensors'))
        processing=dict(size=cfg['size'],resize='sem_yolos.data.letterbox, PIL bilinear, RGB114 padding',
                        processor=read(checkpoint/'preprocessor_config.json'),nms=None,
                        candidate_limit=mc['num_detection_tokens'],ap_score_floor=0.,
                        operating_threshold=threshold,threshold_source='saved threshold.json',
                        semantics='softmax over defects + no-object; foreground argmax excludes no-object')
        precision='fp32'; checkpoint_config=mc
    elif family=='ssd':
        rc=read(run/'run_config.json'); cfg=rc['config'];labels={int(k):v for k,v in rc['external_labels'].items()}
        identity=rc['data_fingerprint'];selection=read(run/'best_selection.json')
        if selection['data_fingerprint']!=identity or rc['selection_split']!='val': raise ValueError('SSD selection provenance mismatch')
        saved=read(run/'best_validation_metrics.json')
        # Existing fixed localization diagnostic, not a tuned deployment threshold.
        threshold=saved['class_agnostic_localization']['confidence_threshold']
        files=[checkpoint] if checkpoint.is_file() else []
        processing=dict(size=cfg['size'],resize_mode=cfg['resize_mode'],resize='sem_ssd.preprocessing.transform, RGB /255',
            nms='class-aware batched_nms on decoded normalized boxes before clipping',nms_iou=cfg['nms_iou'],
            ap_score_floor=cfg['score_floor'],pre_nms_topk=cfg['pre_nms_topk'],max_detections=cfg['max_detections'],
            operating_threshold=threshold,threshold_source='saved fixed class-agnostic diagnostic; not calibrated',
            semantics='softmax with SSD background index0 and defects indices1..4; per-class SSD emission')
        precision=rc['precision'];checkpoint_config=rc
    else: raise ValueError(family)
    if set(labels.values())!=set(mapping.values()): raise ValueError('Model/dataset category mismatch')
    if sorted(labels)!=list(range(len(labels))): raise ValueError('Expected contiguous model foreground IDs')
    if identity!=dataset_report['declared_split_fingerprint']: raise ValueError('Checkpoint differs from reviewed split manifest')
    return dict(family=family,checkpoint=str(checkpoint),checkpoint_available=bool(files),
        checkpoint_files=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in files],
        checkpoint_configuration=checkpoint_config,pipeline_configuration=cfg,external_labels=labels,
        source_categories=mapping,background_index=len(labels) if family=='yolos' else 0,
        declared_checkpoint_fingerprint=identity,preprocessing_postprocessing=processing,precision=precision,
        prediction_path=str(resolve(config['predictions'])),prediction_sha256=sha(resolve(config['predictions'])))


class Adapter:
    def __init__(self,config,metadata,device='cpu',inference=False):
        self.config=config;self.metadata=metadata;self.cfg=metadata['pipeline_configuration']
        self.names=[metadata['external_labels'][i] for i in sorted(metadata['external_labels'])]
        self.source_ids={v:int(k) for k,v in metadata['source_categories'].items()}
        self.device=torch.device(device);self.family=config['family'];self.model=None
        self.processor=None
        if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        if metadata['precision']!='fp32': raise ValueError('Adapter currently requires baseline fp32; no silent precision change')
        if self.family=='yolos':
            from transformers import YolosImageProcessor
            self.processor=YolosImageProcessor.from_pretrained(metadata['checkpoint'],local_files_only=True)
            self.processor.do_resize=False;self.processor.do_pad=False
            if inference:
                from sem_yolos.runtime import load_model
                self.model,self.processor=load_model(Path(metadata['checkpoint']),self.device)
        elif inference:
            if not metadata['checkpoint_available']: raise FileNotFoundError('Trained SSD best.pt is absent; use --mode saved or transfer trained checkpoint')
            from sem_ssd.model import create_model
            saved=torch.load(metadata['checkpoint'],map_location='cpu',weights_only=False)
            if saved['data_fingerprint']!=metadata['declared_checkpoint_fingerprint'] or saved['config']!=self.cfg:
                raise ValueError('SSD checkpoint configuration/fingerprint disagrees with saved run')
            selection=read(resolve(config['run'])/'best_selection.json')
            if saved['stage']!=selection['stage'] or saved['epoch']!=selection['epoch']:
                raise ValueError('SSD checkpoint is not saved best selection')
            self.model,_,_=create_model(self.cfg,pretrained=False)
            self.model.load_state_dict(saved['model'],strict=True);self.model.to(self.device)
        if self.model is not None: self.model.eval()

    def preprocess(self,info,annotations,image_root):
        with Image.open(image_path(image_root,info)) as f: image=f.convert('RGB')
        boxes=[a['bbox'] for a in annotations]
        if self.family=='yolos':
            from sem_yolos.data import letterbox,inverse_boxes
            canvas,transformed,meta=letterbox(image,boxes,self.cfg['size'])
            encoded=self.processor(images=canvas,do_resize=False,return_tensors='pt')
            tensor=encoded['pixel_values'][0]
            xyxy=transformed.clone();xyxy[:,2:]+=xyxy[:,:2]
            roundtrip=inverse_boxes(xyxy,meta)
            display=tensor.clone()
            if self.processor.do_normalize:
                display=display*torch.tensor(self.processor.image_std)[:,None,None]+torch.tensor(self.processor.image_mean)[:,None,None]
            if self.processor.do_rescale: display=display/self.processor.rescale_factor
            display=display.permute(1,2,0).round().clamp(0,255).byte().numpy()
        else:
            from sem_ssd.preprocessing import transform,inverse
            tensor,normalized,meta,_=transform(image,boxes,self.cfg['size'],self.cfg['resize_mode'])
            xyxy=normalized*self.cfg['size'];roundtrip=inverse(normalized,meta,clamp=False)
            display=tensor.permute(1,2,0).mul(255).round().clamp(0,255).byte().numpy()
        original=torch.tensor(boxes,dtype=torch.float32).reshape(-1,4);original[:,2:]+=original[:,:2]
        error=float((original-roundtrip).abs().max()) if len(boxes) else 0.
        if error>1e-3: raise ValueError(f'Coordinate round-trip failed: {error}')
        return dict(tensor=tensor,display=Image.fromarray(display),boxes_xyxy=xyxy.tolist(),meta=meta,
                    roundtrip_max_error_pixels=error,image=image)

    def row(self,identifier,box,scores,stages):
        values=[float(v) for v in scores]
        fg=values[:-1] if self.family=='yolos' else values[1:]
        bg=values[-1] if self.family=='yolos' else values[0]
        idx=max(range(len(fg)),key=lambda i:fg[i])
        return dict(candidate_id=int(identifier),bbox=[float(v) for v in box],full_scores=values,
            foreground_argmax=self.names[idx],foreground_score=fg[idx],gap_score=fg[self.names.index('gap')],
            background_score=bg,background_dominates=bg>=max(fg),stages=stages,
            final_any=any(v['operating_final'] for v in stages.values()),gap_final=stages['gap']['operating_final'])

    def detection(self,image_id,candidate_id,class_index,box,score):
        return dict(image_id=image_id,candidate_id=int(candidate_id),detection_id=f'{image_id}:{candidate_id}:{class_index}',
                    class_name=self.names[class_index],category_id=self.source_ids[self.names[class_index]],
                    bbox=[float(v) for v in box],score=float(score))

    @torch.inference_mode()
    def capture(self,info,prepared):
        if self.model is None: raise RuntimeError('Inference model not loaded')
        threshold=self.metadata['preprocessing_postprocessing']['operating_threshold']
        tensor=prepared['tensor'].unsqueeze(0).to(self.device);meta=prepared['meta']
        if self.family=='yolos':
            from sem_yolos.runtime import decode
            output=self.model(pixel_values=tensor)
            probabilities=output.logits[0].softmax(-1).cpu()
            final=decode(output,[meta],[info['id']])
            rows=[];baseline=[]
            for query,(p,prob) in enumerate(zip(final,probabilities)):
                cls=int(p['category_id']);score=float(p['score']);valid=p['bbox'][2]>0 and p['bbox'][3]>0
                stages={name:dict(confidence_pass=i==cls and score>=threshold,pre_nms_topk=None,nms=None,
                    detection_limit=None,baseline_final=i==cls,operating_final=i==cls and score>=threshold,
                    removal_reason='not_foreground_argmax' if i!=cls else 'below_operating_threshold' if score<threshold else None)
                    for i,name in enumerate(self.names)}
                # Baseline preserves zero-area decoded query boxes; do not silently change it.
                row=self.row(query,p['bbox'],prob.tolist(),stages);row['positive_area']=valid
                row['model_box_normalized_cxcywh']=output.pred_boxes[0,query].cpu().tolist()
                rows.append(row)
                baseline.append(self.detection(info['id'],query,cls,p['bbox'],score))
            return rows,baseline,dict(internal_replay_agrees=True,candidate_count=len(rows),
                                     score_order=self.names+['no_object'],truncation='fixed number of model queries; no NMS or score filter in saved baseline AP predictions')
        from sem_ssd.model import raw_forward,decode
        from sem_ssd.preprocessing import inverse
        pred,anchors=raw_forward(self.model,tensor)
        boxes=self.model.match_prior.convert_to_boxes(pred['boxes'].float(),anchors.float())[0]
        scores=pred['scores'].float().softmax(-1)[0]
        if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all(): raise ValueError('Nonfinite SSD raw output')
        stages,selected=ssd_lineage(boxes,scores,self.cfg,threshold,self.names)
        pixels=inverse(boxes.cpu(),meta);xywh=pixels.clone();xywh[:,2:]-=xywh[:,:2]
        baseline=[]
        for anchor,cls in selected:
            b=xywh[anchor].tolist()
            if b[2]<=0 or b[3]<=0:
                stages[anchor][self.names[cls]]['baseline_final']=False
                stages[anchor][self.names[cls]]['operating_final']=False
                stages[anchor][self.names[cls]]['removal_reason']='zero_area_after_baseline_clipping'
                continue
            baseline.append(self.detection(info['id'],anchor,cls,b,scores[anchor,cls+1]))
        official=decode(self.model,pred,anchors)[0]
        official_boxes=inverse(official.boxes.cpu(),meta);official_boxes[:,2:]-=official_boxes[:,:2]
        expected=[self.detection(info['id'],-1,int(c)-1,b.tolist(),s) for b,c,s in zip(official_boxes,official.labels,official.scores) if b[2]>0 and b[3]>0]
        from core import compare_predictions
        equality=compare_predictions(baseline,expected,{info['id']},1e-5,1e-6)
        if not equality['agrees']: raise RuntimeError(f'SSD lineage replay disagrees with unchanged upstream decode: {equality}')
        score_cpu=scores.cpu().tolist()
        rows=[self.row(i,xywh[i].tolist(),score_cpu[i],stages[i]) for i in range(len(boxes))]
        normalized=boxes.cpu().tolist()
        for i,row in enumerate(rows):row['model_box_normalized_xyxy']=normalized[i]
        return rows,baseline,dict(internal_replay_agrees=True,candidate_count=len(rows),score_order=['background']+self.names,
              truncation='all decoded anchors captured; class floor -> per-class top-k -> batched NMS -> global max -> clip and discard zero area')


def ssd_lineage(boxes,scores,cfg,threshold,names):
    from torchvision.ops import batched_nms
    lineage=[{} for _ in range(len(boxes))];parts=[];classes=[];values=[]
    for c,name in enumerate(names,1):
        passed=torch.where(scores[:,c]>cfg['score_floor'])[0]
        k=min(cfg['pre_nms_topk'],len(passed))
        v,order=scores[passed,c].topk(k);top=passed[order]
        passed_set=set(passed.tolist());top_set=set(top.tolist())
        for i in range(len(boxes)):
            lineage[i][name]=dict(confidence_pass=i in passed_set,pre_nms_topk=i in top_set,nms=False,
                 detection_limit=False,baseline_final=False,operating_final=False,
                 removal_reason='below_score_floor' if i not in passed_set else 'per_class_topk' if i not in top_set else 'unknown')
        parts.extend(top.tolist());classes.extend([c]*k);values.extend(v.unbind())
    if not parts: return lineage,[]
    indices=torch.tensor(parts,device=boxes.device);labels=torch.tensor(classes,device=boxes.device)
    keep=batched_nms(boxes[indices],torch.stack(values),labels,cfg['nms_iou']).tolist()
    keep_set=set(keep);limited=keep[:cfg['max_detections']];limited_set=set(limited)
    for j,(anchor,c) in enumerate(zip(parts,classes)):
        s=lineage[anchor][names[c-1]];s['nms']=j in keep_set;s['detection_limit']=j in limited_set
        s['baseline_final']=j in limited_set;s['operating_final']=j in limited_set and float(scores[anchor,c])>=threshold
        s['removal_reason']='nms' if j not in keep_set else 'global_detection_limit' if j not in limited_set else 'below_operating_threshold' if not s['operating_final'] else None
    return lineage,[(parts[j],classes[j]-1) for j in limited]
