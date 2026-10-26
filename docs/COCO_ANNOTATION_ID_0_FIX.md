# Correction: annotation ID 0 in COCO validation AP

The original Label Studio COCO export has a `multi_bridge` annotation with ID `0` in image ID `0`. Image and category IDs are not the issue. `pycocotools.COCOeval` stores a matched ground-truth **annotation** ID in `dtMatches` and uses `0` as the unmatched sentinel. Consequently, this annotation's matches were mishandled in AP calculations.

`prepare_data.py` now writes positive, unique prepared annotation IDs and preserves the original ID in `source_annotation_id`. `sem_yolos.metrics.coco_ap` independently renumbers the annotations in memory before calling COCOeval, protecting evaluations of any older prepared files. The raw `data/raw/result.json` is unchanged. Split assignments and the dataset fingerprint are unchanged.

## Corrected saved evaluations

The saved `predictions.json` files were reevaluated with the fixed AP code. Only COCO AP values in the validation `metrics.json` files changed. Detection TP/FP/FN, precision/recall/F1, image classification, threshold choices, and test AP did not change.

| Run | Historical validation mAP@.50:.95 | Corrected | Historical validation AP@.50 | Corrected | Historical multi_bridge AP@.50:.95 | Corrected |
|---|---:|---:|---:|---:|---:|---:|
| Natural 20 views | 23.08% | **23.84%** | 31.07% | **32.97%** | 46.01% | **49.05%** |
| Weighted 20 views | 16.89% | **20.54%** | 32.77% | **38.95%** | 40.82% | **55.39%** |

The test annotations have no ID `0`, so test AP remains unchanged: natural mAP 22.99% and weighted mAP 22.73%. The previous test comparisons of threshold-dependent detection and image classification also remain unchanged.

## Historical training selection

The saved `metrics_A_*.json` and `metrics_B_*.json` files were calculated during training with the old COCOeval input. Their validation AP curves cannot be reconstructed for every epoch because per-epoch predictions/checkpoints were not retained. `best/selection.json` and the `best/` weights are therefore **historical selections**, not guaranteed to be the best under corrected AP. Do not quote their historical epoch AP as corrected.

The corrected validation AP of checkpoints that are still available was measured by running them on the 16 validation images:

| Run | Historically selected `best/` | Other saved stage checkpoint | Corrected mAP@.50:.95 |
|---|---|---|---:|
| Natural | Stage A epoch 7; corrected mAP **23.84%** | `stage_B_best/` (historically Stage B epoch 16) | **25.71%** |
| Weighted | Stage B epoch 8; corrected mAP **20.54%** | `stage_A_best/` (historically Stage A epoch 5) | **15.97%** |

For the natural run, the saved Stage B checkpoint outranks the historically selected `best/` under corrected validation AP. This does not establish which of *all* historical epochs would now be best. The stored test result still belongs to the original `best/` checkpoint. The alternative Stage B checkpoint has not been threshold-calibrated or test-evaluated as a replacement. The weighted saved checkpoint comparison still favors its historical `best/`.

Future training runs will use positive annotation IDs and corrected AP for checkpoint selection. Existing model weights, raw images, raw annotations, split assignments, and stored prediction files were not changed by this correction.
