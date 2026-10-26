"""Focused geometry checks for manual-review artifacts."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/review_annotations.py"
SPEC = importlib.util.spec_from_file_location("review_annotations", SCRIPT)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


class ReviewArtifactTests(unittest.TestCase):
    def test_cropped_box_and_source_pixels_near_boundary(self):
        image = Image.new("RGB", (60, 40), (12, 34, 56))
        record = {"split": "train", "image_id": 1, "key": "one"}
        annotation = {"id": 3, "category_id": 1, "bbox": [1, 2, 5, 4]}
        with tempfile.TemporaryDirectory() as directory:
            entry = review.crop_artifacts(image, record, annotation,
                                          Path(directory), 2, 320)
            self.assertEqual(entry["crop_origin"], [0, 0])
            self.assertEqual(entry["original_bbox_xywh"], [1, 2, 5, 4])
            self.assertGreater(entry["display_scale"], 1)
            plain = Image.open(Path(directory) / entry["paths"]["unmarked"])
            marked = Image.open(Path(directory) / entry["paths"]["marked"])
            self.assertEqual(plain.size, tuple(entry["crop_dimensions"]))
            self.assertEqual(plain.getpixel((1, 2)), (12, 34, 56))
            self.assertNotEqual(marked.getpixel((1, 2)), plain.getpixel((1, 2)))

    def test_invalid_box_geometry(self):
        self.assertIsNone(review.valid_box([0, 0, 5, 4], 60, 40))
        self.assertIn("outside", review.valid_box([59, 0, 2, 4], 60, 40))
        self.assertIn("positive", review.valid_box([0, 0, 0, 4], 60, 40))
        self.assertIn("non-finite", review.valid_box([float("nan"), 0, 1, 1], 60, 40))

    def test_multiple_boxes_and_empty_panel(self):
        image = Image.new("RGB", (90, 70), "black")
        base = {"split": "train", "image_id": 2, "filename": "sample.png",
                "existing_status": "unknown", "warnings": []}
        boxes = [{"id": 10, "category_id": 0, "bbox": [5, 8, 9, 6]},
                 {"id": 11, "category_id": 1, "bbox": [40, 30, 8, 7]}]
        marked = review.panel(image, dict(base, annotations=boxes), boxes, "TEST")
        self.assertNotEqual(marked.getpixel((5, marked.height - 70 + 8)), (0, 0, 0))
        self.assertNotEqual(marked.getpixel((40, marked.height - 70 + 30)), (0, 0, 0))
        empty = review.panel(image, dict(base, annotations=[]), [], "TEST")
        self.assertEqual(empty.getpixel((20, empty.height - 70 + 20)), (0, 0, 0))

    def test_training_letterbox_inverse(self):
        try:
            from sem_yolos.data import letterbox, inverse_boxes
        except ImportError:
            self.skipTest("Torch preprocessing unavailable")
        source = [[3, 7, 11, 5], [20, 13, 7, 8]]
        _, canvas_boxes, meta = letterbox(Image.new("RGB", (90, 45)), source, 512)
        canvas_xyxy = canvas_boxes.clone()
        canvas_xyxy[:, 2:] += canvas_xyxy[:, :2]
        inverse = inverse_boxes(canvas_xyxy, meta).tolist()
        for actual, (x, y, width, height) in zip(inverse, source):
            for got, expected in zip(actual, [x, y, x + width, y + height]):
                self.assertAlmostEqual(got, expected, places=3)

    def test_gap_and_representative_selection(self):
        def make(image_id, categories):
            return {"split": "train", "image_id": image_id, "path": Path("image.png"),
                    "annotations": [{"category_id": c} for c in categories],
                    "valid_annotations": [{"category_id": c} for c in categories]}
        records = [make(0, [1]), make(1, [1]), make(2, [0]),
                   make(3, [2]), make(4, [3]), make(5, [])]
        records.append(dict(make(6, [1]), split="val"))
        self.assertEqual([r["image_id"] for r in review.select_augmentation_records(records, False)],
                         [0, 1, 2, 3, 4, 5])
        self.assertEqual(len(review.select_augmentation_records(records, True)), 6)


if __name__ == "__main__":
    unittest.main()
