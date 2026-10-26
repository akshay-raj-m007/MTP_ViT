# Dataset report

```json
{
  "images": 95,
  "boxes": 69,
  "instances": {
    "multi_bridge": 10,
    "line_collapse": 21,
    "gap": 13,
    "bridge": 25
  },
  "class_presence": {
    "multi_bridge": 10,
    "line_collapse": 20,
    "gap": 7,
    "bridge": 22
  },
  "split_distributions": {
    "train": {
      "multi_bridge": 6,
      "no_defect": 26,
      "line_collapse": 14,
      "gap": 3,
      "bridge": 14
    },
    "val": {
      "multi_bridge": 2,
      "no_defect": 5,
      "line_collapse": 3,
      "gap": 2,
      "bridge": 4
    },
    "test": {
      "multi_bridge": 2,
      "no_defect": 5,
      "line_collapse": 3,
      "gap": 2,
      "bridge": 4
    }
  },
  "split_box_distributions": {
    "train": {
      "multi_bridge": 6,
      "line_collapse": 15,
      "gap": 6,
      "bridge": 17
    },
    "val": {
      "multi_bridge": 2,
      "line_collapse": 3,
      "gap": 4,
      "bridge": 4
    },
    "test": {
      "multi_bridge": 2,
      "line_collapse": 3,
      "gap": 3,
      "bridge": 4
    }
  }
}
```

35 empty images assumed clean from COCO and supplied dataset description; review metadata unavailable.

fig34 is clean (no_defect) by explicit user confirmation; 36 clean images total, with zero detection boxes.

Near-duplicate candidates are review hints, not inferred source groups.

Source groups supplied: False

Full dimensions, hashes and near-duplicate candidates: data/prepared/label_studio/validation_report.json and split_manifest.json. Small originals include 81x81 pixels; upscaling cannot recover detail. No test predictions were computed.


Validation: {
  "errors": [],
  "unique_image_ids": true,
  "unique_annotation_ids": true,
  "crowd_annotations": 0,
  "ignored_annotations": 0,
  "exact_duplicate_groups": [],
  "near_duplicate_cross_split_candidates": 13,
  "boxes_shorter_than_4_pixels_at_512": 0,
  "minimum_box_short_side_at_512": 11.31940298507463
}

Near-duplicate candidates: 31. See NEAR_DUPLICATE_REVIEW.md before interpreting source-independent generalization.
