import unittest
from collections import Counter
import torch
from PIL import Image,ImageDraw
from torchvision.transforms.v2 import functional as F
from sem_yolos.common import ROOT,read
from sem_yolos.data import CocoDetectionDataset,sample_weights,letterbox
from sem_yolos.multi_augment import geometry,appearance,validate_recipes

class MultiAugmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.config=read(ROOT/'configs/pipeline_config_20_views.json')
        cls.recipes=cls.config['augmentation_recipes']
    def dataset(self,training=True,split='train'):
        return CocoDetectionDataset(ROOT,ROOT/'data/prepared/label_studio'/(split+'.json'),None,128,training,
                                    view_mode='original_plus_recipes',recipes=self.recipes)
    def test_twenty_views_and_source_weights(self):
        ds=self.dataset();self.assertEqual(len(ds),1260);self.assertEqual(ds.views_per_image,20)
        counts=Counter((ds.view_info(i)[0],ds.view_name(i)) for i in range(len(ds)))
        self.assertEqual(len(counts),1260);self.assertEqual(set(counts.values()),{1})
        weights,report=sample_weights(ds);self.assertEqual(len(weights),1260)
        self.assertEqual(weights,weights[:63]*20)
        self.assertEqual(report['image_class_frequencies']['gap'],3)
        self.assertEqual(len(self.dataset(False,'val')),16)
        with self.assertRaises(ValueError):validate_recipes([{'name':'bad','crop':True}])
    def test_known_stretch_rotation_and_content(self):
        image=Image.new('RGB',(100,80),'black');ImageDraw.Draw(image).rectangle((10,20,39,29),fill='white')
        resized,b,labels=geometry(image,[[10,20,30,10]],[1],dict(scale_x=1.2))
        self.assertEqual(resized.size,(120,80));torch.testing.assert_close(b,torch.tensor([[12.,20.,36.,10.]]))
        rotated,b,labels=geometry(image,[[10,20,30,10]],[1],dict(angle=90.))
        self.assertEqual(rotated.size,(80,100));torch.testing.assert_close(b,torch.tensor([[20.,60.,10.,30.]]),atol=1e-4,rtol=1e-5)
        self.assertEqual(int((F.to_image(rotated)[0]>240).sum()),300)
        self.assertEqual(labels,[1])
    def test_full_frame_rotation_no_clipping(self):
        image=Image.new('RGB',(101,79),'white')
        rotated,b,_=geometry(image,[[0,0,101,79]],[0],dict(angle=12.))
        self.assertGreater(rotated.width,101);self.assertGreater(rotated.height,79)
        bright=(F.to_image(rotated)[0]>240).nonzero()
        x,y,w,h=b[0].tolist()
        self.assertGreaterEqual(float(bright[:,1].min()),x-1)
        self.assertLessEqual(float(bright[:,1].max()),x+w+1)
        self.assertGreaterEqual(float(bright[:,0].min()),y-1)
        self.assertLessEqual(float(bright[:,0].max()),y+h+1)
    def test_shear_pixel_alignment_and_box_extent(self):
        image=Image.new('RGB',(101,79),'black')
        ImageDraw.Draw(image).rectangle((10,20,39,29),fill='white')
        for recipe in [dict(shear_x=8.),dict(shear_y=-8.),dict(shear_x=5.,angle=12.)]:
            transformed,b,labels=geometry(image,[[10,20,30,10]],[2],recipe)
            bright=(F.to_image(transformed)[0]>240).nonzero()
            self.assertGreater(len(bright),200)
            x,y,w,h=b[0].tolist()
            self.assertGreaterEqual(float(bright[:,1].min()),x-1)
            self.assertLessEqual(float(bright[:,1].max()),x+w+1)
            self.assertGreaterEqual(float(bright[:,0].min()),y-1)
            self.assertLessEqual(float(bright[:,0].max()),y+h+1)
            self.assertEqual(labels,[2])
        _,b,_=geometry(image,[[10,20,30,10]],[2],dict(shear_x=8.))
        self.assertAlmostEqual(float(b[0,2]),30+10*__import__('math').tan(__import__('math').radians(8)),places=3)
    def test_scale_is_not_cancelled_by_letterbox(self):
        image=Image.new('RGB',(100,80));views=[]
        for zoom in [.8,1.15]:
            im,b,_=geometry(image,[[10,20,30,10]],[1],dict(zoom=zoom))
            _,b,_=letterbox(im,b,512);views.append(b[0,2].item())
        self.assertGreater(views[1],views[0]*1.4)
    def test_noise_sigma_in_normalized_units(self):
        torch.manual_seed(42)
        im=appearance(Image.new('RGB',(256,256),(128,128,128)),dict(noise_std=.03))
        value=F.to_image(im).float()/255.
        self.assertAlmostEqual(float(value.std()),.03,delta=.001)
        torch.testing.assert_close(value[0],value[1]);torch.testing.assert_close(value[1],value[2])
    def test_all_training_sources_all_recipes_preserve_targets(self):
        ds=self.dataset()
        for i in range(len(ds)):
            source,_=ds.view_info(i);expected=ds.anns_by_image[ds.images[source]['id']]
            canvas,b,labels,_=ds.transformed(i)
            self.assertEqual(canvas.size,(128,128));self.assertEqual(len(b),len(expected))
            self.assertEqual(labels,[a['category_id'] for a in expected])
            if b.numel():
                self.assertTrue(bool((b[:,2:]>0).all()))
                self.assertTrue(bool((b[:,:2]>=-1e-3).all()))
                self.assertTrue(bool((b[:,0]+b[:,2]<=128+1e-3).all()))
                self.assertTrue(bool((b[:,1]+b[:,3]<=128+1e-3).all()))

if __name__=='__main__':unittest.main()
