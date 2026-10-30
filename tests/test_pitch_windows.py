import random
import unittest
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.v2 import functional as F
from sem_yolos.common import ROOT, read
from sem_yolos.data import CocoDetectionDataset, inverse_boxes, letterbox
from sem_yolos.multi_augment import geometry
from sem_yolos.windows import crop_boxes, crop_window, merge_predictions, tile_windows

PREPARED = ROOT/'data/prepared/label_studio_v3_pitch32'
CONFIG = read(ROOT/'configs/pipeline_config_pitch32.json')


def dataset(split, training=False, tiled=False):
    return CocoDetectionDataset(ROOT, PREPARED/f'{split}.json', None, CONFIG['size'], training=training,
                                view_mode=CONFIG['training_view_mode'] if training else 'random_augmented',
                                recipes=CONFIG['augmentation_recipes'] if training else None, native=True,
                                tile_overlap=CONFIG['tile_overlap'] if tiled else None)


class WindowTests(unittest.TestCase):
    def test_crop_holds_the_box_and_stays_inside_the_image(self):
        rng = random.Random(0)
        for _ in range(300):
            box = [rng.uniform(0, 1100), rng.uniform(0, 1100), rng.uniform(5, 100), rng.uniform(5, 300)]
            x, y, w, h = crop_window(1400, 1500, 640, [box], rng)
            self.assertTrue(0 <= x and x+w <= 1400 and 0 <= y and y+h <= 1500 and (w, h) == (640, 640))
            self.assertTrue(x <= box[0] and y <= box[1] and box[0]+box[2] <= x+w and box[1]+box[3] <= y+h)

    def test_box_longer_than_the_window_is_cropped_inside_it(self):
        x, y, w, h = crop_window(300, 1500, 640, [[10, 100, 20, 900]], random.Random(1))
        self.assertTrue(100 <= y and y+h <= 1000 and (x, w, h) == (0, 300, 640))

    def test_small_image_is_not_cropped(self):
        self.assertEqual(crop_window(200, 300, 640, [], random.Random(2)), (0, 0, 200, 300))

    def test_crop_boxes_clips_and_drops(self):
        boxes = [[10, 10, 20, 20], [90, 10, 20, 20], [99, 50, 20, 20]]
        kept, labels = crop_boxes(boxes, [0, 1, 2], (0, 0, 100, 100))
        self.assertEqual(labels, [0, 1])
        self.assertEqual(kept[1], [90, 10, 10, 20])

    def test_tiles_cover_every_pixel_and_add_the_whole_image(self):
        for width, height in [(1470, 1452), (660, 300), (641, 641), (1124, 1124)]:
            tiles = tile_windows(width, height, 640, 320)
            self.assertEqual(tiles[-1], (0, 0, width, height))
            mask = torch.zeros(height, width, dtype=torch.bool)
            for x, y, w, h in tiles[:-1]:
                self.assertTrue(w <= 640 and h <= 640 and x+w <= width and y+h <= height)
                mask[y:y+h, x:x+w] = True
            self.assertTrue(mask.all())
        self.assertEqual(tile_windows(640, 200, 640, 320), [(0, 0, 640, 200)])

    def test_every_box_up_to_the_overlap_is_whole_in_some_tile(self):
        rng = random.Random(3)
        tiles = tile_windows(1470, 1452, 640, 320)[:-1]
        for _ in range(500):
            w, h = rng.uniform(5, 320), rng.uniform(5, 320)
            x, y = rng.uniform(0, 1470-w), rng.uniform(0, 1452-h)
            self.assertTrue(any(a <= x and b <= y and x+w <= a+c and y+h <= b+d for a, b, c, d in tiles))

    def test_merge_removes_duplicates_only_on_tiled_images(self):
        box = dict(image_id=1, category_id=0, bbox=[10, 10, 20, 20], score=.9)
        other = dict(box, category_id=1, score=.8)
        plain = dict(box, image_id=2)
        merged = merge_predictions([box, dict(box, score=.5), other, plain, plain], {1})
        self.assertEqual(sum(p['image_id'] == 1 for p in merged), 2)
        self.assertEqual(sum(p['image_id'] == 2 for p in merged), 2)
        self.assertEqual(max(p['score'] for p in merged if p['image_id'] == 1 and p['category_id'] == 0), .9)

    def test_small_image_keeps_its_scale(self):
        canvas, boxes, meta = letterbox(Image.new('RGB', (100, 50)), [[10, 5, 20, 10]], 640, upscale=False)
        self.assertEqual((meta['scale_x'], meta['scale_y'], float(boxes[0, 2])), (1., 1., 20.))
        self.assertEqual(canvas.size, (640, 640))

    def test_inverse_applies_the_tile_offset(self):
        _, boxes, meta = letterbox(Image.new('RGB', (640, 640)), [[10, 5, 20, 10]], 640, upscale=False)
        meta.update(offset_x=300, offset_y=200)
        xyxy = torch.cat((boxes[:, :2], boxes[:, :2]+boxes[:, 2:]), 1)
        self.assertEqual(inverse_boxes(xyxy, meta)[0].tolist(), [310., 205., 330., 215.])


class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = dataset('train', training=True)

    def test_all_views_present_and_scale_recipes_stay_within_ten_percent(self):
        self.assertEqual(self.train.views_per_image, 21)
        self.assertEqual(len(self.train), 66*21)
        for recipe in CONFIG['augmentation_recipes']:
            self.assertTrue(.9 <= recipe.get('zoom', 1) <= 1.1 and .9 <= recipe.get('scale_x', 1) <= 1.1, recipe['name'])

    def test_every_view_of_a_defect_image_shows_a_defect_at_native_scale(self):
        n = len(self.train.images)
        for source, info in enumerate(self.train.images):
            originals = self.train.anns_by_image[info['id']]
            for view in range(1, 21):
                canvas, boxes, labels, meta = self.train.transformed(source+view*n)
                self.assertEqual(canvas.size, (640, 640))
                if originals: self.assertGreater(len(labels), 0, info['file_name'])
                else: self.assertEqual(len(labels), 0)
                if len(boxes):
                    self.assertTrue(bool((boxes[:, 2:] > 0).all()))
                    self.assertTrue(bool((boxes[:, :2] >= -1e-3).all()) and bool((boxes[:, :2]+boxes[:, 2:] <= 640+1e-3).all()))
                self.assertLessEqual(max(meta['scale_x'], meta['scale_y']), 1.)

    def test_boxes_follow_pixels_through_crop_recipe_and_padding(self):
        rng = random.Random(4)
        for source, info in enumerate(self.train.images):
            for ann in self.train.anns_by_image[info['id']]:
                x, y, w, h = [round(v) for v in ann['bbox']]
                mask = Image.new('RGB', (info['width'], info['height']), 'black')
                ImageDraw.Draw(mask).rectangle((x, y, x+w-1, y+h-1), fill='white')
                for recipe in CONFIG['augmentation_recipes']:
                    window = crop_window(info['width'], info['height'], 640, [[x, y, w, h]], rng)
                    kept, labels = crop_boxes([[x, y, w, h]], [0], window)
                    cropped = mask.crop((window[0], window[1], window[0]+window[2], window[1]+window[3]))
                    out, boxes, _ = geometry(cropped, kept, labels, recipe)
                    canvas, placed, _ = letterbox(out, boxes, 640, upscale=False)
                    lit = (F.to_image(canvas)[0] > 180).nonzero()
                    bx, by, bw, bh = placed[0].tolist()
                    found = (float(lit[:, 1].min()), float(lit[:, 0].min()), float(lit[:, 1].max())+1, float(lit[:, 0].max())+1)
                    self.assertTrue(all(abs(a-b) <= 2.5 for a, b in zip(found, (bx, by, bx+bw, by+bh))), (info['file_name'], recipe['name'], found))

    def test_tiled_evaluation_maps_every_ground_truth_box_back(self):
        for split in ('val', 'test'):
            data = dataset(split, tiled=True)
            self.assertGreater(len(data), len(data.images))
            recovered = {}
            for index in range(len(data)):
                source = data.tiles[index][0]; info = data.images[source]
                canvas, boxes, labels, meta = data.transformed(index)
                for box in boxes.tolist():
                    xyxy = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
                    recovered.setdefault(info['id'], []).append(inverse_boxes([xyxy], meta)[0].tolist())
            for info in data.images:
                for ann in data.anns_by_image[info['id']]:
                    x, y, w, h = ann['bbox']; want = [x, y, x+w, y+h]
                    best = min(max(abs(a-b) for a, b in zip(want, got)) for got in recovered[info['id']])
                    self.assertLess(best, 1.5, info['file_name'])


if __name__ == '__main__':
    unittest.main()
