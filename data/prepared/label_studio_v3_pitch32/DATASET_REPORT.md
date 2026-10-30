# Dataset report

```json
{
  "images": 95,
  "boxes": 70,
  "instances": {
    "bridge": 25,
    "gap": 14,
    "line_collapse": 21,
    "multi_bridge": 10
  },
  "class_presence": {
    "bridge": 22,
    "gap": 7,
    "line_collapse": 20,
    "multi_bridge": 10
  },
  "split_distributions": {
    "train": {
      "gap": 5,
      "line_collapse": 14,
      "bridge": 15,
      "multi_bridge": 6,
      "no_defect": 26
    },
    "val": {
      "bridge": 4,
      "line_collapse": 3,
      "multi_bridge": 3,
      "gap": 1,
      "no_defect": 5
    },
    "test": {
      "multi_bridge": 1,
      "bridge": 3,
      "line_collapse": 3,
      "no_defect": 5,
      "gap": 1
    }
  },
  "split_box_distributions": {
    "train": {
      "gap": 9,
      "line_collapse": 14,
      "bridge": 18,
      "multi_bridge": 6
    },
    "val": {
      "bridge": 4,
      "line_collapse": 3,
      "multi_bridge": 3,
      "gap": 4
    },
    "test": {
      "multi_bridge": 1,
      "bridge": 3,
      "line_collapse": 4,
      "gap": 1
    }
  }
}
```

Images without boxes are clean (no_defect) as exported by the annotator; they carry zero detection boxes.

fig34 is clean (no_defect) by explicit user confirmation; 36 clean images total, with zero detection boxes.

Near-duplicate candidates are review hints, not inferred source groups.

Source groups supplied: True. Split plan: data/splits/split_v2.json

Full dimensions, hashes and near-duplicate candidates: data/prepared/label_studio_v3_pitch32/validation_report.json and split_manifest.json. Small originals include 81x81 pixels; upscaling cannot recover detail. No test predictions were computed.


Validation: {
  "errors": [],
  "unique_image_ids": true,
  "unique_annotation_ids": true,
  "crowd_annotations": 0,
  "ignored_annotations": 0,
  "exact_duplicate_groups": [],
  "near_duplicate_cross_split_candidates": 0,
  "boxes_shorter_than_4_pixels_at_512": 0,
  "minimum_box_short_side_at_512": 13.158586866724763
}

Near-duplicate candidates: 31. See NEAR_DUPLICATE_REVIEW.md before interpreting source-independent generalization.
