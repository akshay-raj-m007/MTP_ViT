import unittest
from collections import Counter
import torch
from torch.utils.data import RandomSampler
from transformers import YolosImageProcessor
from sem_yolos.common import ROOT
from sem_yolos.data import CocoDetectionDataset,collate_fn,sample_weights

class TrainingViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor=YolosImageProcessor.from_pretrained('hustvl/yolos-tiny',local_files_only=True)
        cls.processor.do_resize=False;cls.processor.do_pad=False
    def dataset(self,training=True,mode='original_plus_augmented',split='train'):
        return CocoDetectionDataset(ROOT,ROOT/'data/prepared/label_studio_v2'/(split+'.json'),self.processor,128,training,
                    dict(hflip=1.,vflip=0.,brightness=0.,contrast=0.),view_mode=mode)
    def test_natural_epoch_original_and_augmented_once(self):
        ds=self.dataset();self.assertEqual(len(ds),132);self.assertEqual(len(ds.images),66)
        counts=Counter()
        for index in RandomSampler(ds,generator=torch.Generator().manual_seed(42)):
            source,augmented=ds.view_info(index);counts[(ds.images[source]['id'],augmented)]+=1
        self.assertEqual(len(counts),132);self.assertEqual(set(counts.values()),{1})
    def test_original_matches_deterministic_and_augmented_keeps_targets(self):
        ds=self.dataset();plain=self.dataset(training=False)
        indices=[next(i for i,im in enumerate(ds.images) if not im['classes']),
                 next(i for i,im in enumerate(ds.images) if len(ds.anns_by_image[im['id']])>=2)]
        for i in indices:
            original,augmented=ds[i],ds[i+len(ds.images)];reference=plain[i]
            torch.testing.assert_close(original['pixel_values'],reference['pixel_values'])
            torch.testing.assert_close(original['labels']['boxes'],reference['labels']['boxes'])
            self.assertEqual(original['view_kind'],'original');self.assertEqual(augmented['view_kind'],'augmented')
            self.assertEqual(original['image_id'],augmented['image_id'])
            torch.testing.assert_close(original['labels']['class_labels'],augmented['labels']['class_labels'])
            self.assertEqual(original['labels']['boxes'].shape,augmented['labels']['boxes'].shape)
            batch=collate_fn([original,augmented]);self.assertEqual(tuple(batch['pixel_values'].shape),(2,3,128,128))
    def test_sampling_uses_source_frequencies_and_expands_weights(self):
        ds=self.dataset();baseline=self.dataset(mode='random_augmented')
        weights,report=sample_weights(ds);base,base_report=sample_weights(baseline)
        self.assertEqual(weights,base+base)
        self.assertEqual(report['image_class_frequencies'],base_report['image_class_frequencies'])
        self.assertEqual(report['expected_clean_fraction'],base_report['expected_clean_fraction'])
        self.assertEqual(report['presentations_per_epoch'],132)
    def test_validation_and_baseline_are_not_expanded(self):
        val=self.dataset(training=False,split='val');self.assertEqual(len(val),16)
        self.assertTrue(all(not val.view_info(i)[1] for i in range(len(val))))
        baseline=self.dataset(mode='random_augmented');self.assertEqual(len(baseline),66)
        self.assertTrue(all(baseline.view_info(i)[1] for i in range(len(baseline))))
        with self.assertRaises(ValueError):self.dataset(mode='invalid')

if __name__=='__main__':unittest.main()
