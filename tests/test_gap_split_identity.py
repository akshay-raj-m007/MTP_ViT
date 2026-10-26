"""The prepared split identity must change when gap assignments change."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sem_yolos.common import prepared_fingerprint, read


EXPECTED = {
    "train": {"baea786d-fig14.jpg", "d40dc8e8-fig2.jpg", "79c680ce-fig29.jpg"},
    "val": {"a3c2b033-fig26.jpg", "d799ade3-fig45.jpg"},
    "test": {"b5e1d95a-fig35.jpg", "add3881f-fig91.jpg"},
}


class GapSplitIdentityTests(unittest.TestCase):
    def test_gap_images_and_annotations_follow_requested_split(self):
        prepared = ROOT / "data/prepared/label_studio"
        manifest = read(prepared / "split_manifest.json")
        self.assertEqual(manifest["split_fingerprint"], prepared_fingerprint(prepared))
        self.assertEqual(manifest["fingerprint"],
                         "b3aa0c38134c1bf18ed47502308a6740423d3544328ac59662f5572991e6b587")
        image_ids = set()
        expected_images = {"train": 63, "val": 16, "test": 16}
        expected_boxes = {"train": 44, "val": 13, "test": 12}
        expected_gap_boxes = {"train": 6, "val": 4, "test": 3}
        for split in ("train", "val", "test"):
            data = read(prepared / f"{split}.json")
            gaps = {a["image_id"] for a in data["annotations"] if a["category_id"] == 1}
            names = {Path(im["file_name"]).name for im in data["images"] if im["id"] in gaps}
            self.assertEqual(names, EXPECTED[split])
            self.assertEqual(len(data["images"]), expected_images[split])
            self.assertEqual(len(data["annotations"]), expected_boxes[split])
            self.assertEqual(sum(a["category_id"] == 1 for a in data["annotations"]),
                             expected_gap_boxes[split])
            ids = {im["id"] for im in data["images"]}
            self.assertFalse(image_ids & ids)
            image_ids |= ids
            self.assertTrue(all(a["image_id"] in ids and a["id"] > 0 for a in data["annotations"]))
        self.assertEqual(len(image_ids), 95)

    def test_fingerprint_changes_when_split_content_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = Path(directory)
            (prepared / "split_manifest.json").write_text(json.dumps({"fingerprint": "same-source"}))
            for split in ("train", "val", "test"):
                (prepared / f"{split}.json").write_text(json.dumps({"images": [], "annotations": []}))
            before = prepared_fingerprint(prepared)
            (prepared / "train.json").write_text(json.dumps({"images": [{"id": 1}], "annotations": []}))
            self.assertNotEqual(before, prepared_fingerprint(prepared))


if __name__ == "__main__":
    unittest.main()
