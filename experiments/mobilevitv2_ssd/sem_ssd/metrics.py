"""Same whole-split COCO definitions as sem_yolos.metrics, without importing it."""
import contextlib
import copy
import io
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from .io import CLASSES


def localization_recall(data, predictions, confidence=0.3, iou_threshold=0.5):
    """Score-ordered one-to-one matching ignoring class, after detector NMS."""
    by_image = {}
    for ann in data['annotations']:
        by_image.setdefault(ann['image_id'], []).append(ann['bbox'])
    matched = {image_id:set() for image_id in by_image}
    def iou(a, b):
        ax,ay,aw,ah=a; bx,by,bw,bh=b
        intersection=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
        return intersection/(aw*ah+bw*bh-intersection) if intersection else 0.
    for pred in sorted(predictions,key=lambda p:p['score'],reverse=True):
        if pred['score'] < confidence: continue
        image_id=pred['image_id']
        available=[(iou(pred['bbox'],box),i) for i,box in enumerate(by_image.get(image_id,[]))
                   if i not in matched[image_id]]
        if available:
            overlap,index=max(available)
            if overlap >= iou_threshold: matched[image_id].add(index)
    support=sum(len(boxes) for boxes in by_image.values())
    tp=sum(len(indices) for indices in matched.values())
    return dict(recall=tp/support if support else None,matched_boxes=tp,support=support,
                confidence_threshold=confidence,iou_threshold=iou_threshold,
                matching='score_ordered_one_to_one_class_agnostic_after_class_aware_nms')


def coco_ap(data, predictions):
    data = copy.deepcopy(data)
    for index, ann in enumerate(data['annotations'], 1):
        ann['id'] = index  # evaluator-local sentinel protection; raw IDs remain intact
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(); gt.dataset = data; gt.createIndex()
        if predictions:
            dt = gt.loadRes(predictions)
        else:
            dt = COCO(); dt.dataset = dict(data, annotations=[]); dt.createIndex()
        ev = COCOeval(gt, dt, 'bbox')
        ev.params.imgIds = sorted(im['id'] for im in data['images'])
        ev.evaluate(); ev.accumulate(); ev.summarize()
    def mean(values):
        values = values[values >= 0]
        return float(values.mean()) if len(values) else None
    precision = ev.eval['precision']
    return dict(map=mean(precision[:,:,:,0,2]), map50=mean(precision[0,:,:,0,2]),
                per_class_ap={CLASSES[c]:dict(ap=mean(precision[:,:,k,0,2]),
                    ap50=mean(precision[0,:,k,0,2]), support=int(sum(a['category_id']==c for a in data['annotations'])))
                    for k,c in enumerate(ev.params.catIds)})
