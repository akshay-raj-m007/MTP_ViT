"""Whole-split COCO AP plus explicitly matched detection/image metrics."""
import contextlib
import io
from collections import defaultdict, Counter
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from .common import CLASSES


def iou(a,b):
    x,y,w,h=a; X,Y,W,H=b
    intersection=max(0,min(x+w,X+W)-max(x,X))*max(0,min(y+h,Y+H)-max(y,Y))
    return intersection/max(1e-12,w*h+W*H-intersection)


def prf(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0.; r=tp/(tp+fn) if tp+fn else 0.
    return dict(precision=p,recall=r,f1=2*p*r/(p+r) if p+r else 0.,support=tp+fn,tp=tp,fp=fp,fn=fn)


def fixed_metrics(coco,predictions,threshold):
    gt=defaultdict(list); pred=defaultdict(list)
    for a in coco['annotations']: gt[a['image_id']].append(a)
    allowed={im['id'] for im in coco['images']}
    for a in predictions:
        if a['image_id'] not in allowed: raise ValueError('Prediction image ID outside split')
        if a['score']>=threshold: pred[a['image_id']].append(a)
    totals=[[0,0,0] for _ in CLASSES]; clean=clean_fp=clean_bad=defective=detected=0
    confusion=np.zeros((5,5),dtype=int); multi=False; presence=[[0,0,0] for _ in CLASSES]
    for im in coco['images']:
        if not im.get('evaluation_eligible',True): continue
        g=gt[im['id']]; ps=sorted(pred[im['id']],key=lambda p:p['score'],reverse=True); used=set()
        for p in ps:
            c=p['category_id']
            candidates=[(iou(p['bbox'],a['bbox']),j) for j,a in enumerate(g) if j not in used and a['category_id']==c]
            overlap,j=max(candidates,default=(0.,-1))
            if overlap>=.5: used.add(j); totals[c][0]+=1
            else: totals[c][1]+=1
        for j,a in enumerate(g):
            if j not in used: totals[a['category_id']][2]+=1
        truth={a['category_id'] for a in g}; guessed={p['category_id'] for p in ps}
        for c in range(4):
            presence[c][0]+=int(c in truth and c in guessed)
            presence[c][1]+=int(c not in truth and c in guessed)
            presence[c][2]+=int(c in truth and c not in guessed)
        if not truth: clean+=1;clean_fp+=len(ps);clean_bad+=bool(ps)
        else: defective+=1;detected+=bool(ps)
        if len(truth)>1: multi=True
        else: confusion[next(iter(truth),4),ps[0]['category_id'] if ps else 4]+=1
    tp,fp,fn=map(sum,zip(*totals))
    result=dict(threshold=threshold,detection=prf(tp,fp,fn),per_class_detection={n:prf(*v) for n,v in zip(CLASSES,totals)},
                clean_images=clean,false_positive_boxes_per_clean_image=clean_fp/clean if clean else None,
                fraction_clean_with_false_detection=clean_bad/clean if clean else None,
                binary_defective_recall=detected/defective if defective else None,
                clean_specificity=1-clean_bad/clean if clean else None,clean_false_alarm_rate=clean_bad/clean if clean else None)
    if multi:
        values={n:prf(*v) for n,v in zip(CLASSES,presence)}
        result['image_classification']=dict(mode='multilabel_presence',per_class=values,macro_f1=float(np.mean([v['f1'] for v in values.values()])))
    else:
        values={}
        for c,n in enumerate(CLASSES+['no_defect']):
            values[n]=prf(int(confusion[c,c]),int(confusion[:,c].sum()-confusion[c,c]),int(confusion[c].sum()-confusion[c,c]))
        supported=[v['recall'] for v in values.values() if v['support']]
        result['image_classification']=dict(mode='single_label',labels=CLASSES+['no_defect'],confusion_matrix=confusion.tolist(),per_class=values,
                                            macro_f1=float(np.mean([v['f1'] for v in values.values()])),balanced_accuracy=float(np.mean(supported)) if supported else None)
    return result


def coco_ap(coco,predictions):
    eligible={im['id'] for im in coco['images'] if im.get('evaluation_eligible',True)}
    dataset=dict(coco,images=[im for im in coco['images'] if im['id'] in eligible],annotations=[a for a in coco['annotations'] if a['image_id'] in eligible])
    predictions=[p for p in predictions if p['image_id'] in eligible]
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO();gt.dataset=dataset;gt.createIndex()
        if predictions: dt=gt.loadRes(predictions)
        else:
            dt=COCO();dt.dataset=dict(dataset,annotations=[]);dt.createIndex()
        ev=COCOeval(gt,dt,'bbox');ev.params.imgIds=sorted(eligible)
        ev.evaluate();ev.accumulate();ev.summarize()
    def valid_mean(x):
        values=x[x>=0];return float(values.mean()) if len(values) else None
    per={}
    for k,c in enumerate(ev.params.catIds):
        per[CLASSES[c]]=dict(ap=valid_mean(ev.eval['precision'][:,:,k,0,2]),ap50=valid_mean(ev.eval['precision'][0,:,k,0,2]),
                            support=int(sum(a['category_id']==c for a in dataset['annotations'])))
    return dict(map=valid_mean(ev.eval['precision'][:,:,:,0,2]),map50=valid_mean(ev.eval['precision'][0,:,:,0,2]),per_class_ap=per)


def evaluate_predictions(coco,predictions,threshold=.5):
    return dict(**coco_ap(coco,predictions),**fixed_metrics(coco,predictions,threshold))


def select_threshold(coco,predictions):
    # Predefined grid; tie-break to the higher threshold. Validation only at caller.
    grid=[i/100 for i in range(5,96,5)]
    scored=[(fixed_metrics(coco,predictions,t)['detection']['f1'],t) for t in grid]
    score,threshold=max(scored)
    return dict(threshold=threshold,objective='validation detection F1 at IoU 0.5',grid=grid,validation_f1=score)
