# Code map

Two isolated pipelines read the same prepared data in `data/prepared/label_studio_v2/`: relabeled export with the grouped split. `data/prepared/label_studio/` keeps the original-label baseline that the existing runs and diagnostics configs use. Windows commands are shown with `$py = .\venv\Scripts\python.exe` for YOLOS and `$ssd = .\outputs\mobilevitv2_ssd\env\Scripts\python.exe` for SSD.

## Data preparation (shared)
- `scripts/prepare_data.py` reads `data/raw/result_labeled.json` and `data/raw/images/`. It strips the extra Label Studio upload prefix. It writes `train/val/test.json`, `split_manifest.json`, `class_mapping.json`, `validation_report.json`, `NEAR_DUPLICATE_REVIEW.md` and `DATASET_REPORT.md` into the output directory.
- Command: `& $py .\scripts\prepare_data.py`. The split comes from `--assign data/splits/split_v2.json` (filename to split, with same-defect groups; see `data/splits/SPLIT_V2.md`). It fails if a group crosses splits or if any image is missing or unknown.
- The old baseline: `--source data/raw/result.json --output data/prepared/label_studio --assign ''` (keeps its saved manifest, matched by filename).
- Annotation review: `& $py .\scripts\review_annotations.py`.

## YOLOS-Tiny (`scripts/`, `sem_yolos/`)
| Step | Command |
|---|---|
| Train natural | `& $py .\scripts\train_yolos.py --config .\configs\pipeline_config_20_views.json --sampling natural --output .\outputs\runs\<run>` |
| Train weighted | the same command with `--config .\configs\pipeline_config_20_views_weighted_balanced.json --sampling weighted` |
| Val + threshold | `& $py .\scripts\evaluate.py --model .\outputs\runs\<run>\best --split val --select-threshold --output .\outputs\runs\<run>\validation` |
| Test | `& $py .\scripts\evaluate.py --model ...\best --split test --output ...\final_test` |
| Inference | `& $py .\scripts\infer_yolos.py <image> --model ...\best --output <dir>` |
| Review predictions | `& $py .\scripts\draw_predictions.py --predictions ...\final_test\predictions.json --split test --threshold <thr> --output <dir>` (GT beside predictions ≥0.05, plus `gt_review.csv`: found / wrong_class / poorly_localised / missed per GT box; works on SSD `predictions.json` too) |
| 768 input | `configs/pipeline_config_20_views_768.json` and `..._weighted_balanced_768.json` (identical to the 512 configs except `size`) |

- **Config keys:** `size` (letterbox size), the stage epochs and learning rates, `patience` (early stopping, active), `sampling_power`/`sampling_cap` (weighted mode only), `training_view_mode` plus `augmentation_recipes` (the 20 views), and `augmentation` (used only in `random_augmented` mode).
- **Outputs per run:** `best/` (HF checkpoint, `selection.json`, `threshold.json`), `stage_A_best/`, `stage_B_best/`, `last/` (with `training_state.pt`), `metrics_{A,B}_NNN.json`, `sampling_report.json`, `device.json`.
- **Current runs:** `outputs/runs/unweighted_gap_resplit_01`, `outputs/runs/weighted_gap_resplit_01`.

## SSD-MobileViTv2 (`experiments/mobilevitv2_ssd/`)
- Entry: `scripts/experiment.py {setup,check-setup,inspect,preview,smoke,train,overfit,validate,test}`.
- Train: `& $ssd .\experiments\mobilevitv2_ssd\scripts\experiment.py train --config .\experiments\mobilevitv2_ssd\configs\natural.json --seed 42 --device cuda --precision fp32 --output outputs/mobilevitv2_ssd/<run>`. Use `weighted.json` for the weighted run.
- Validate: `... validate --model outputs/mobilevitv2_ssd/<run>/best.pt --device cuda --precision fp32`.
- Test: `... test --model outputs/mobilevitv2_ssd/<run>/best.pt --device cuda --precision fp32 --output outputs/mobilevitv2_ssd/<run>/eval_test` (COCO AP plus `predictions.json`; no threshold). `SemDataset` still refuses the test split unless `final_test=True`, which only this command passes.
- **Config keys:** `size` and `resize_mode` (resize or letterbox), `sampling`, `empty_negatives`, `early_stopping` (off), and the NMS settings `score_floor`, `nms_iou`, `pre_nms_topk`, `max_detections`.
- **Outputs:** `best.pt`, `stage_{A,B}_best.pt`, `last.pt`, `best_validation_metrics.json`, `best_validation_predictions.json`, `metrics_*.json`.
- **Current runs:** `natural_seed42_retry_20261027_230508`, `weighted_seed42`.

## Diagnostics
- Gallery: `diagnostics/prediction_gallery.py capture-ssd|build` (commands are in `diagnostics/PREDICTION_GALLERY.md`). Output goes to `outputs/prediction_gallery/`.
- Gap failure analysis: `diagnostics/sem_defects/diagnose.py {inspect,run,combine,review-summary} --config diagnostics/sem_defects/configs/<model>.json`. Output goes to `outputs/diagnostics_sem/`.
