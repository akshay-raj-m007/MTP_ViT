import unittest
import json
import tempfile
from pathlib import Path
import torch
from PIL import Image
from transformers import YolosImageProcessor
from scripts.prepare_data import mapping,split_records
from sem_yolos.common import ROOT,read,resolve_image,CATEGORIES
from sem_yolos.data import letterbox,inverse_boxes,augment,CocoDetectionDataset,sample_weights
from sem_yolos.metrics import evaluate_predictions,fixed_metrics


class PipelineTests(unittest.TestCase):
    def test_noncontiguous_mapping(self):
        self.assertEqual(mapping([{'id':91,'name':'gap'},{'id':8,'name':'bridge'},{'id':0,'name':'no_defect'}]),{91:1,8:0})
    def test_windows_path_and_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'images').mkdir();(root/'images'/'prefix-x.jpg').touch()
            self.assertEqual(resolve_image(root,r'..\media\prefix-x.jpg').name,'prefix-x.jpg')
            (root/'images'/'sub').mkdir();(root/'images'/'sub'/'prefix-x.jpg').touch()
            with self.assertRaises(ValueError): resolve_image(root,'prefix-x.jpg')
    def test_letterbox_roundtrip(self):
        image=Image.new('RGB',(301,107));boxes=torch.tensor([[20.,10.,30.,40.],[0.,0.,301.,107.]])
        canvas,b,meta=letterbox(image,boxes,512)
        self.assertEqual(canvas.size,(512,512));b[:,2:]+=b[:,:2]
        original=boxes.clone();original[:,2:]+=original[:,:2]
        torch.testing.assert_close(inverse_boxes(b,meta),original,atol=1e-4,rtol=1e-5)
    def test_known_resize_and_processor_coordinates(self):
        canvas,b,meta=letterbox(Image.new('RGB',(100,80)),[[10,20,30,10]],200)
        torch.testing.assert_close(b,torch.tensor([[20.,60.,60.,20.]]))
        processor=YolosImageProcessor.from_pretrained('hustvl/yolos-tiny',local_files_only=True)
        encoded=processor(images=canvas,annotations={'image_id':7,'annotations':[dict(bbox=b[0].tolist(),category_id=1,area=1200,iscrowd=0)]},do_resize=False,do_pad=False,return_tensors='pt')
        torch.testing.assert_close(encoded['labels'][0]['boxes'],torch.tensor([[.25,.35,.3,.1]]))
        self.assertEqual(tuple(encoded['pixel_values'].shape),(1,3,200,200))
    def test_known_flip(self):
        image=Image.new('RGB',(100,80));_,b,labels=augment(image,[[10,20,30,10]],[2],dict(hflip=1.,vflip=1.,brightness=0.,contrast=0.))
        torch.testing.assert_close(b,torch.tensor([[60.,50.,30.,10.]]));self.assertEqual(labels,[2])
        _,b,_=augment(image,[],[],dict(hflip=1.,vflip=1.));self.assertEqual(tuple(b.shape),(0,4))
    def test_prepared_leakage(self):
        manifest=read(ROOT/'data/prepared/label_studio/split_manifest.json');seen={};ids=set()
        for im in manifest['images']:
            self.assertNotIn(im['id'],ids);ids.add(im['id'])
            for key in (im['sha256'],im['leakage_group']):
                if key in seen:self.assertEqual(seen[key],im['split'])
                seen[key]=im['split']
            if not im['evaluation_eligible']: self.assertEqual(im['split'],'train')
        self.assertEqual(len(ids),95)
    def test_fig34_is_clean(self):
        manifest=read(ROOT/'data/prepared/label_studio/split_manifest.json')
        image=next(im for im in manifest['images'] if im['file_name'].endswith('c5329556-fig34.jpg'))
        self.assertEqual(image['classes'],[])
        self.assertEqual(image['annotation_status'],'user_confirmed_clean')
        self.assertTrue(image['evaluation_eligible'])
        self.assertEqual(image['split'],'train')
        self.assertEqual(sum(not im['classes'] for im in manifest['images']),36)
    def test_duplicate_and_source_groups(self):
        records=[dict(id=i,file_name=f'{i}.jpg',sha256='same' if i<2 else str(i),classes=[i%4],evaluation_eligible=True) for i in range(12)]
        result=split_records(records,42,{'1.jpg':'source','2.jpg':'source'})
        self.assertEqual(len({x['split'] for x in result[:3]}),1)
    def test_processor_empty_and_multi(self):
        processor=YolosImageProcessor.from_pretrained('hustvl/yolos-tiny',local_files_only=True)
        processor.do_resize=False;processor.do_pad=False
        ds=CocoDetectionDataset(ROOT,ROOT/'data/prepared/label_studio/train.json',processor,128)
        for count in [0,2]:
            index=next(i for i,im in enumerate(ds.images) if (len(ds.anns_by_image[im['id']])==0 if count==0 else len(ds.anns_by_image[im['id']])>=2))
            item=ds[index];self.assertEqual(tuple(item['pixel_values'].shape),(3,128,128))
            n=len(ds.anns_by_image[item['image_id']]);self.assertEqual(tuple(item['labels']['boxes'].shape),(n,4))
            self.assertEqual(int(item['labels']['image_id'].item()),item['image_id'])
        weights,report=sample_weights(ds);self.assertEqual(len(weights),len(ds));self.assertLessEqual(max(weights),3.)
    def fixture(self):
        return dict(info={},images=[dict(id=1,width=100,height=100),dict(id=2,width=100,height=100)],categories=CATEGORIES,
                    annotations=[dict(id=1,image_id=1,category_id=0,bbox=[10,10,20,20],area=400,iscrowd=0)])
    def test_perfect_metrics(self):
        d=self.fixture();p=[dict(image_id=1,category_id=0,bbox=[10,10,20,20],score=.9)]
        m=evaluate_predictions(d,p,.5);self.assertAlmostEqual(m['map'],1.);self.assertAlmostEqual(m['map50'],1.)
        json.dumps(m,allow_nan=False);self.assertEqual(m['detection']['f1'],1.);self.assertEqual(m['clean_specificity'],1.)
    def test_zero_annotation_id_does_not_corrupt_coco_ap(self):
        d=self.fixture();d['annotations'][0]['id']=0
        p=[dict(image_id=1,category_id=0,bbox=[10,10,20,20],score=.9)]
        m=evaluate_predictions(d,p,.5)
        self.assertAlmostEqual(m['map'],1.)
        self.assertAlmostEqual(m['map50'],1.)
        self.assertEqual(d['annotations'][0]['id'],0)

    def test_prepared_annotation_ids_are_positive(self):
        for split in ('train','val','test'):
            d=read(ROOT/'data/prepared/label_studio'/(split+'.json'))
            ids=[a['id'] for a in d['annotations']]
            self.assertTrue(all(isinstance(i,int) and i>0 for i in ids))
            self.assertEqual(len(ids),len(set(ids)))

    def test_duplicates_wrong_classes_clean_fp(self):
        d=self.fixture();p=[dict(image_id=1,category_id=0,bbox=[10,10,20,20],score=.9)]
        m=fixed_metrics(d,p+p+[dict(image_id=2,category_id=1,bbox=[1,1,5,5],score=.8)],.5)
        self.assertEqual((m['detection']['tp'],m['detection']['fp'],m['detection']['fn']),(1,2,0))
        self.assertEqual(m['false_positive_boxes_per_clean_image'],1.)
        p[0]['category_id']=1;m=fixed_metrics(d,p,.5)
        self.assertEqual((m['detection']['tp'],m['detection']['fp'],m['detection']['fn']),(0,1,1))
        m=evaluate_predictions(d,[],.5);self.assertEqual(m['map'],0.)
    def test_multilabel(self):
        d=self.fixture();d['annotations'].append(dict(id=2,image_id=1,category_id=1,bbox=[50,50,10,10],area=100,iscrowd=0))
        self.assertEqual(fixed_metrics(d,[],.5)['image_classification']['mode'],'multilabel_presence')

if __name__=='__main__': unittest.main()
