# YOLOS-Tiny SEM detection: Label Studio pipeline

Four foreground classes: `0 bridge`, `1 gap`, `2 line_collapse`, `3 multi_bridge`.
`no_defect` is an image-level threshold outcome, never a foreground box category.
Raw input is **data/raw/images/** plus **data/raw/result.json**. Old Roboflow folders/checkpoints/results are preserved and never used by the new commands.

## Environment and setup (PowerShell)

```powershell
Set-Location 'D:\aksha\MTP\annotate_image.v4-train-test.coco'
$py = '.\venv\Scripts\python.exe'
& $py --version
& $py -m pip install -r requirements.txt
```

The existing environment was verified with Python 3.14.2, PyTorch 2.13.0, torchvision 0.28.0 and Transformers 5.16.1. It is CPU-only. Requirements pin these installed versions; they are a verified environment snapshot, not a claim of compatibility with older Python releases. For CUDA, install matching CUDA-enabled torch/torchvision builds appropriate to your hardware before training. The code probes CUDA/BF16 support, uses FP16 with scaling when needed, and uses full precision on CPU. CUDA was not tested here. Cached `hustvl/yolos-tiny` weights were available; training is local-cache-only unless `--allow-download` is supplied.

## 1. Validate and prepare

```powershell
& $py .\scripts\prepare_data.py
```

Outputs go to `data/prepared/label_studio/`. This resolves Windows-style paths by exact basename, validates all boxes/files, maps source IDs by name, and records hashes. It never clips raw invalid boxes. Crowd/ignore or unknown box labels fail explicitly. This project has a COCO export; native JSON conversion is not needed and unsupported native inputs fail explicitly.

Existing assignments are reused. Any change in annotations, image bytes or supplied groups requires review and `--resplit`. Categories are identical across splits. Exact pixel duplicates and explicitly supplied source groups cannot cross splits. Optional groups JSON maps an exact exported basename (including prefix), or `data/raw/images/<basename>`, to an arbitrary source-group string:

```json
{"43e08f23-fig5.jpg": "paper-A-acquisition-1"}
```

```powershell
& $py .\scripts\prepare_data.py --groups .\source_groups.json --resplit
```

Supply the same group file on later preparation runs. Do not infer source groups from filenames. See `data/prepared/label_studio/NEAR_DUPLICATE_REVIEW.md`: 31 similarity candidates, 13 crossing current splits. These are not proven duplicates; without source review, source-independent generalization remains unverified. If confirmed, add groups and resplit **before training**, then use fresh runs.

Actual data: 95 images, 69 boxes. Split: **63 / 16 / 16**. There are **36 clean (`no_defect`) images**, including `c5329556-fig34.jpg`, explicitly confirmed clean by the user. It retains its existing training assignment and has zero detection boxes. Clean counts are **26 train / 5 validation / 5 test**. Its manifest status is `user_confirmed_clean`; the other 35 use `assumed_clean_coco`, since COCO does not preserve review/Choices metadata. Both statuses count as clean. See `DATASET_REPORT.md`.

## 2. Inspect training augmentation previews

```powershell
& $py .\scripts\preview_data.py --output .\outputs\checks\previews_new
& $py .\scripts\preview_data.py --deterministic --output .\outputs\checks\previews_deterministic
```

Already generated and inspected: `outputs/checks/augmented_previews/` (one image per class and a clean image). Preview output directories must be new. Augmentation is on-the-fly; files are not multiplied for training. The default configuration uses configurable flips and mild brightness/contrast. Optional noise is disabled. This default configuration has no crop, rotation, blur, Mixup, CutMix, mosaic or elastic deformation. For rotation, shear and scaling recipes, use the twenty-view configuration described below.

512x512 is an initial experiment. Images span 81x81 through 1024x1024, including non-square originals. Letterboxing records actual rounded x/y scales and padding. The processor then rescales/normalizes once, without resizing/padding again. Prediction coordinates are unpadded and unscaled before metrics. Upscaling tiny originals recovers no detail; examine small-defect retention before a separate tiling experiment.

## 3. Focused tests and model smoke test

```powershell
& $py -m unittest discover -s tests -v
& $py .\scripts\smoke_test.py --output .\outputs\checks\smoke_new --size 512
```

The smoke test uses two real training examples (empty and multiple boxes), two optimizer steps, and a five-image validation subset. It does not use test images. Successful saved report: `outputs/checks/smoke_512_verified/report.json`. Its mAP=0 is from an essentially untrained four-class head and is not a performance result. An earlier smoke attempt completed computation but failed JSON serialization; the NumPy scalar bug was fixed and the complete test rerun successfully.

## 4. Natural baseline training

Edit `configs/pipeline_config.json` for canvas, budgets, learning rates, workers, augmentation, and sampling settings. Default microbatch 1, accumulation 8, workers 0. Save a separate config file to run another experiment.

```powershell
& $py .\scripts\train_yolos.py --output .\outputs\runs\natural_01 --sampling natural --device auto
```

Stage A: heads only, 15 epochs maximum, LR 1e-4. Stage B: start at best Stage A, unfreeze final two encoder blocks and final normalization, up to 40 epochs; heads LR 5e-5, encoder LR 5e-6. AdamW, weight decay 1e-4, clipping 1.0, short warmup and cosine scheduling. The optimizer is rebuilt at the transition. Tokens/positions remain frozen. Trainable names/counts are saved and printed.

Each epoch evaluates the entire validation split. `map` (COCO AP@0.5:0.95), larger is better, selects the checkpoint and controls stage early stopping (patience 8). Overall `best/` retains the best of both stages. Full loss components use standard YOLOS loss. No test selection. No external experiment reporting.

The dedicated loop avoids Trainer sampler/output-version assumptions. Only one process/GPU is supported. Fresh output directories are mandatory unless explicitly resuming:

```powershell
& $py .\scripts\train_yolos.py --output .\outputs\runs\natural_01 --sampling natural --resume --device auto
```

Resume restores model, optimizer, scheduler, scaler, stage/epoch, best metric, patience and RNG state from the last completed epoch. Mid-epoch progress is discarded. Resume requires identical configuration, sampling and prepared dataset fingerprint. Workers=0 is the tested reproducibility baseline; cross-device and worker-count reproducibility is not guaranteed.

## 5. Optional weighted comparison

```powershell
& $py .\scripts\train_yolos.py --output .\outputs\runs\weighted_01 --sampling weighted --device auto
```

Weights use training-image class presence, not box counts: maximum `(largest image frequency / class image frequency)^0.5` among classes present, capped at 3. All empty targets use explicit configurable `background_weight=1.0`, including the user-confirmed clean example. This is a chosen sampling weight, not a guaranteed neutral clean proportion. The run records weighted/natural expected empty fractions and reports total clean exposure and its assumed-clean/user-confirmed components. Validation/test always evaluate each example once. Keep the same prepared splits when comparing sampling policies.

## 6. Validation and threshold selection

```powershell
& $py .\scripts\evaluate.py --model .\outputs\runs\natural_01\best --split val --select-threshold --output .\outputs\runs\natural_01\validation
```

Selects one global threshold on the predefined grid 0.05, 0.10, ..., 0.95 by detection F1 at IoU 0.5 (ties choose the higher threshold). Saves `best/threshold.json`. Calibration refuses to overwrite an existing threshold. AP always uses every query, not the display threshold. No NMS is applied to YOLOS outputs; duplicate detections count as false positives.

Reports COCO AP/AP50/per-class AP and support; one-to-one detection precision/recall/F1; clean-image FP boxes/image and false-alarm fraction; separate image classification metrics. Single-label images use the top accepted detection or `no_defect`; genuinely multi-class data switches to multilabel presence metrics. No-object probability is not a calibrated whole-image cleanliness score.

## 7. One final test evaluation

Only after all model/augmentation/sampling decisions are fixed:

```powershell
& $py .\scripts\evaluate.py --model .\outputs\runs\natural_01\best --split test --output .\outputs\runs\natural_01\final_test
```

Applies the saved threshold unchanged. Test threshold selection is rejected. Existing output directories are rejected to avoid overwriting results. Do not repeatedly test candidate models during development. No test evaluation was executed during implementation.

## 8. New-image inference and attention visualization

```powershell
& $py .\scripts\infer_yolos.py .\data\raw\images\43e08f23-fig5.jpg --model .\outputs\runs\natural_01\best --output .\outputs\predictions\example_01 --device auto
& $py .\scripts\vis.py .\data\raw\images\43e08f23-fig5.jpg --model .\outputs\runs\natural_01\best --output .\outputs\predictions\attention_01 --device auto
```

Inference saves original-coordinate detections, derived image class/class presence, and annotated PNG without modifying the input. No accepted detection means only that nothing passed the selected threshold. Attention selects an original unfiltered query and displays its patch attention on the matching padded canvas. It is an inspection aid, not a causal explanation. `scripts/arch_visualization.py` uses the same corrected entry point; original extensive presentation plots are preserved in `archive/scripts/legacy_visualization/` for reference and still point to old checkpoints.

## Files and verification

- `sem_yolos/`: one dataset/augmentation implementation, geometry, metrics, runtime utilities.
- `scripts/prepare_data.py`, `configs/pipeline_config.json`: preparation and configuration.
- `scripts/train_yolos.py`: maintained training command; weighted and dataset v2 files are thin compatibility entry points.
- `scripts/evaluate.py`, `scripts/infer_yolos.py`, `scripts/vis.py`, `scripts/arch_visualization.py`: new-run evaluation and prediction tools.
- `scripts/preview_data.py`, `scripts/smoke_test.py`, `tests/`: checks and previews.
- `DATASET_REPORT.md`, `IMPLEMENTATION_NOTES.md`, `outputs/checks/`: findings, limitations and evidence.

The original implementation did not launch a full training run. Subsequently completed runs are preserved in `outputs/runs/`. Older data and checkpoints are in `archive/`; notes and presentation sources are under `docs/`.


## Optional original plus augmented training experiment

`configs/pipeline_config_original_plus_augmented.json` sets `training_view_mode` to `original_plus_augmented`.
The default (when omitted) is the existing `random_augmented` policy. Existing run configs remain unchanged.

With natural sampling, each epoch visits all 63 originals without random augmentation and one randomly augmented view of each: **126 presentations from 63 source images**. Both views receive deterministic letterboxing and normalization. The original branch bypasses flips/jitter/noise. Augmented views are regenerated on access; no image files or split assignments are added. Validation and test remain single deterministic views.

Sampling weights are computed from unique source images and duplicated across their views. Weighted sampling draws 126 views with replacement, so it does not guarantee every source or both versions appear in a given epoch. Use natural sampling first to test the explicit original-plus-augmented experiment.

```powershell
& $py .\scripts\preview_data.py --config .\configs\pipeline_config_original_plus_augmented.json --output .\outputs\checks\paired_previews_new
& $py .\scripts\train_yolos.py --config .\configs\pipeline_config_original_plus_augmented.json --sampling natural --output .\outputs\runs\natural_original_aug_01 --device auto
& $py .\scripts\evaluate.py --model .\outputs\runs\natural_original_aug_01\best --split val --select-threshold --output .\outputs\runs\natural_original_aug_01\validation
& $py .\scripts\summarize_run.py --run .\outputs\runs\natural_original_aug_01
```

The run records view mode, source count and presentations per epoch in sampling_report.json, plus presentations and optimizer-update opportunities in epoch metrics. With batch 1 / accumulation 8, 126 presentations give 16 optimizer-update opportunities per epoch versus 8 in the original experiment. The separate config retains the 15/40 epoch budgets: this intentionally increases exposure and compute, so it is not an equal-compute comparison. For an approximately update-matched secondary comparison, copy the config and use Stage A=8, Stage B=20 and patience=4; actual exposure still depends on early stopping.

All classes, including clean images, receive both views. Class imbalance ratios therefore remain unchanged; improvement in gap or multi_bridge AP is not guaranteed. Compare validation per-class AP and overall mAP, and keep test data reserved. Start a fresh run from the pretrained model; do not resume the old run with changed configuration.


## Twenty-view augmentation mode

Implemented `configs/pipeline_config_20_views.json`: one original plus 19 views (flips, axis stretches/compressions, rotations, shears, padded scaling, noise levels and combinations). Natural sampling visits all 1,260 views from 63 source images per epoch. Validation and test stay deterministic. Bounding boxes follow the same transforms; no labels are dropped. See [AUGMENTATION_20_VIEWS.md](AUGMENTATION_20_VIEWS.md) for exact recipes, training/preview commands, compute implications and verification.
