import csv
import json
import sys
import unittest
import uuid
import importlib.util
from pathlib import Path

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,OUTPUT,category_mapping,iou,match,gap_search,save_review,read,compare_predictions,new_output


class DiagnosticTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('transformers'),'Transformers absent in this environment')
    def test_yolos_tensor_matches_baseline_dataset(self):
        import torch
        from transformers import YolosImageProcessor
        from adapters import Adapter
        from sem_yolos.data import CocoDetectionDataset
        checkpoint=ROOT/'outputs/runs/unweighted_gap_resplit_01/best'
        processor=YolosImageProcessor.from_pretrained(checkpoint,local_files_only=True)
        processor.do_resize=False;processor.do_pad=False
        cfg=read(checkpoint/'pipeline_config.json')
        data=CocoDetectionDataset(ROOT,ROOT/cfg['prepared']/'val.json',processor,cfg['size'])
        adapter=Adapter.__new__(Adapter);adapter.family='yolos';adapter.cfg=cfg;adapter.processor=processor
        info=data.images[0];prepared=adapter.preprocess(info,data.anns_by_image[info['id']],ROOT)
        self.assertTrue(torch.equal(prepared['tensor'],data[0]['pixel_values']))

    def test_iou(self):
        self.assertEqual(iou([0,0,10,10],[0,0,10,10]),1)
        self.assertEqual(iou([0,0,10,10],[20,0,10,10]),0)
        self.assertAlmostEqual(iou([0,0,10,10],[5,0,10,10]),1/3)
        self.assertEqual(iou([0,0,0,0],[0,0,0,0]),0)

    def test_category_mapping_noncontiguous(self):
        categories=[dict(id=i,name=n) for i,n in zip([3,12,19,80],['bridge','gap','line_collapse','multi_bridge'])]
        self.assertEqual(category_mapping(categories+[dict(id=90,name='no_defect')])[12],'gap')
        self.assertNotIn(90,category_mapping(categories+[dict(id=90,name='no_defect')]))
        with self.assertRaises(ValueError):category_mapping(categories+[dict(id=90,name='unknown')])

    def test_adjacent_gaps_one_prediction(self):
        gt=[dict(id=1,image_id=1,class_name='gap',bbox=[0,0,10,10]),dict(id=2,image_id=1,class_name='gap',bbox=[10,0,10,10])]
        pred=[dict(detection_id='q1',image_id=1,class_name='gap',bbox=[0,0,20,10],score=.9)]
        r=match(gt,pred);self.assertEqual((r['tp'],r['fp'],r['fn']),(1,0,1))
        r=match(gt,pred+[dict(pred[0],detection_id='q2')]);self.assertEqual(r['tp'],2)
        with self.assertRaises(ValueError):match(gt,pred*2)

    def test_empty_and_wrong_class(self):
        gt=[dict(id=1,image_id=1,class_name='gap',bbox=[0,0,10,10])]
        p=dict(detection_id='q',image_id=1,class_name='bridge',bbox=[0,0,10,10],score=.9)
        self.assertEqual(match(gt,[])['fn'],1)
        self.assertEqual(match([],[])['tp'],0)
        self.assertEqual(match([],[p])['fp'],1)
        self.assertEqual(match(gt,[p])['tp'],0)
        self.assertEqual(match(gt,[p],False)['tp'],1)
        self.assertEqual(match(gt,[dict(p,image_id=2,class_name='gap')])['fp'],1)

    def test_missing_candidate(self):
        ann=dict(bbox=[0,0,10,10])
        self.assertIsNone(gap_search(ann,[])['C50'])
        self.assertFalse(gap_search(ann,None)['available'])
        c=dict(candidate_id=1,bbox=[50,50,10,10],foreground_argmax='bridge',gap_score=.01)
        self.assertIsNone(gap_search(ann,[c])['B'])
        self.assertIsNone(gap_search(ann,[c])['C25'])

    def test_candidate_reuse_is_not_detection_matching(self):
        c=dict(candidate_id=1,bbox=[0,0,20,10],foreground_argmax='gap',gap_score=.8)
        self.assertEqual(gap_search(dict(bbox=[0,0,10,10]),[c])['A']['candidate_id'],1)
        self.assertEqual(gap_search(dict(bbox=[10,0,10,10]),[c])['A']['candidate_id'],1)

    def test_preserve_review_and_refuse_overwrite(self):
        out=new_output(OUTPUT/('test_review_'+uuid.uuid4().hex))
        row=dict(model='m',image_id=1,annotation_id=2,measurement=3)
        first=out/'first.csv';save_review(first,[row])
        with first.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        rows[0].update(reviewed='yes',reviewer_notes='Keep my judgement',primary_failure='uncertain')
        with first.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        old=first.read_bytes();second=out/'second.csv';save_review(second,[dict(row,measurement=4)],first)
        with second.open(encoding='utf-8-sig',newline='') as f:review=list(csv.DictReader(f))[0]
        self.assertEqual(review['reviewer_notes'],'Keep my judgement');self.assertEqual(review['measurement'],'4')
        self.assertEqual(first.read_bytes(),old)
        with self.assertRaises(FileExistsError):save_review(first,[row])

    def test_prediction_comparison(self):
        p=dict(image_id=1,category_id=3,bbox=[1,2,3,4],score=.2)
        self.assertTrue(compare_predictions([p],[p],{1})['agrees'])
        self.assertFalse(compare_predictions([p],[],{1})['agrees'])
        self.assertFalse(compare_predictions([dict(p,score=.21)],[p],{1})['agrees'])

    def test_combined_review_rerun_preserves_edits(self):
        from core import write
        from visuals import combined_report
        from PIL import Image
        root=new_output(OUTPUT/('test_combined_'+uuid.uuid4().hex))
        source=root/'source';source.mkdir()
        write(source/'manifest.json',dict(experiment_id='ssd_fixture',mode='saved',dataset=dict(splits={
            s:dict(annotation_sha256=s) for s in ('train','val')})))
        metrics=dict(tp=0,fp=0,fn=1,support=1)
        write(source/'summary.json',dict(gap_detection=metrics,gap_localization_class_agnostic=metrics,
             operating_threshold=.3,coverage='full_validation',missing_evidence=['raw anchors']))
        write(source/'per_gap.json',[dict(image_id=1,annotation_id=2,search=dict(available=False),visuals=dict(panel='panel.png'))])
        Image.new('RGB',(10,10),'white').save(source/'panel.png')
        (source/'index.html').write_text('Fixture',encoding='utf-8')
        save_review(source/'manual_review.csv',[dict(model='ssd_fixture',image_id=1,annotation_id=2,
            reviewer_notes='Existing source review',reviewed='yes',primary_failure='uncertain')])
        first=root/'v1';first.mkdir();combined_report([source],first,['ssd_fixture'])
        with (first/'manual_review.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        self.assertEqual(rows[0]['reviewer_notes'],'Existing source review')
        self.assertEqual(rows[0]['raw_candidate_evidence'],'unknown_not_captured')
        rows[0]['reviewer_notes']='Reviewer edit with comma, and unicode: μm'
        rows[0]['annotation_correct']='uncertain';rows[0]['input_visibility']='uncertain'
        with (first/'manual_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        protected={p:p.read_bytes() for folder in (source,first) for p in folder.iterdir() if p.is_file()}
        second=root/'v2';second.mkdir();combined_report([source],second,['ssd_fixture'],first/'manual_review.csv')
        with (second/'manual_review.csv').open(encoding='utf-8-sig',newline='') as f:revised=list(csv.DictReader(f))[0]
        for key in ('reviewer_notes','reviewed','primary_failure','annotation_correct','input_visibility'):
            self.assertEqual(revised[key],rows[0][key])
        self.assertTrue(all(p.read_bytes()==value for p,value in protected.items()))
        self.assertIn('UNKNOWN',(second/'index.html').read_text(encoding='utf-8'))

    def test_actual_geometry(self):
        import torch
        from PIL import Image
        sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/mobilevitv2_ssd'))
        from sem_yolos.data import letterbox,inverse_boxes
        from sem_ssd.preprocessing import transform,inverse
        im=Image.new('RGB',(113,61));boxes=[[7,9,21,13]]
        _,b,meta=letterbox(im,boxes,512);b[:,2:]+=b[:,:2]
        expected=torch.tensor([[7.,9,28,22]])
        self.assertTrue(torch.allclose(inverse_boxes(b,meta),expected,atol=1e-4))
        for mode in ['resize','letterbox']:
            _,b,meta,_=transform(im,boxes,320,mode)
            self.assertTrue(torch.allclose(inverse(b,meta,clamp=False),expected,atol=1e-4))

    def test_ssd_lineage(self):
        import torch
        from adapters import ssd_lineage
        cfg=dict(score_floor=.05,pre_nms_topk=3,max_detections=1,nms_iou=.5)
        boxes=torch.tensor([[0.,0.,1.,1.],[0.,0.,1.,1.],[2.,0.,3.,1.],[4.,0.,5.,1.]])
        scores=torch.tensor([[.1,.9],[.2,.8],[.3,.7],[.9,.01]])
        stages,final=ssd_lineage(boxes,scores,cfg,.3,['gap'])
        self.assertEqual(final,[(0,0)])
        self.assertEqual(stages[1]['gap']['removal_reason'],'nms')
        self.assertEqual(stages[2]['gap']['removal_reason'],'global_detection_limit')
        self.assertEqual(stages[3]['gap']['removal_reason'],'below_score_floor')
        cfg['pre_nms_topk']=1
        stages,_=ssd_lineage(boxes,scores,cfg,.3,['gap'])
        self.assertEqual(stages[1]['gap']['removal_reason'],'per_class_topk')

    def test_background_mapping(self):
        from adapters import Adapter
        a=Adapter.__new__(Adapter);a.names=['bridge','gap','line_collapse','multi_bridge']
        stages={n:dict(operating_final=False) for n in a.names}
        a.family='yolos';r=a.row(0,[0,0,1,1],[.01,.04,.02,.03,.9],stages)
        self.assertTrue(r['background_dominates']);self.assertEqual(r['foreground_argmax'],'gap');self.assertEqual(r['gap_score'],.04)
        a.family='ssd';r=a.row(0,[0,0,1,1],[.9,.01,.04,.02,.03],stages)
        self.assertTrue(r['background_dominates']);self.assertEqual(r['gap_score'],.04)

    def test_ssd_lineage_matches_official_postprocess(self):
        import torch
        from types import SimpleNamespace
        from adapters import ssd_lineage
        from sem_ssd.upstream import activate
        activate()
        from cvnets.models.detection.ssd import SingleShotMaskDetector
        boxes=torch.tensor([[0.,0.,1.,1.],[.01,.01,1.,1.],[2.,0.,3.,1.],[4.,0.,5.,1.]])
        scores=torch.tensor([[.01,.80,.02,.02,.15],[.01,.70,.10,.09,.10],[.01,.05,.60,.24,.10],[.90,.01,.02,.03,.04]])
        cfg=dict(score_floor=.025,pre_nms_topk=3,max_detections=3,nms_iou=.5)
        names=['bridge','gap','line_collapse','multi_bridge']
        _,selected=ssd_lineage(boxes,scores,cfg,.3,names)
        official=SingleShotMaskDetector.postprocess_detections(SimpleNamespace(conf_threshold=.025,top_k=3,nms_threshold=.5,objects_per_image=3),boxes[None],scores[None])[0]
        self.assertEqual([c+1 for i,c in selected],official.labels.tolist())
        self.assertTrue(torch.equal(torch.stack([boxes[i] for i,c in selected]),official.boxes))
        self.assertTrue(torch.equal(torch.stack([scores[i,c+1] for i,c in selected]),official.scores))


if __name__=='__main__':unittest.main()
