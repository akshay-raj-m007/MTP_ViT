# Twenty training views per image

Use `configs/pipeline_config_20_views.json` with the existing training command. The mode is `original_plus_recipes`. Each source image produces one original view plus one view for each of 19 named recipes. Under natural sampling, every one of these views is visited once per epoch, in shuffled order.

## Views

| Type | Additional views | Parameters |
|---|---:|---|
| Flips | 3 | Horizontal, vertical, both |
| Width/height stretching and compression | 4 | Width 1.2x/0.8x; height 1.2x/0.8x |
| Rotations | 2 | -12 and +12 degrees |
| Shears | 2 | Horizontal +8 degrees; vertical -8 degrees |
| Scaling on a padded reference canvas | 2 | 0.8x and 1.15x |
| Gaussian noise | 3 | Sigma 0.01, 0.03, 0.05 in [0,1] intensity units |
| Combined transforms | 3 | Flips, stretching/scaling, rotations, noise, mild intensity jitter; first two also use shear |
| Original | 1 | Deterministic resize/pad/normalization only |
| **Total** | **20** | 19 augmented views plus original |

Noise uses the same noise field across RGB channels so grayscale SEM images remain grayscale. Sigma refers to the noise before intensity clipping/8-bit quantization and is applied after letterboxing; noise is resampled on access. The three standalone sigmas correspond to approximately 2.55, 7.65 and 12.75 on an 8-bit intensity scale. Geometric parameters are fixed per named recipe; combined-view intensity jitter is random. Symmetric source images may produce visually identical flips: 20 recipe slots do not guarantee 20 different pixel arrays.

## Bounding boxes and preservation of content

Image and boxes use the same torchvision v2 flip, resize, pad, affine and crop operations. For rotation/shear, a conservative padded canvas encloses the transformed original image. Both pixels and boxes receive the same affine parameters. Only padding beyond the transformed source footprint is removed. No source defect is intentionally cropped and no label is silently removed.

The earlier use of separate `rotate(expand=True)` kernels was removed: those kernels disagree on expanded canvas sizes for some shapes, and the empty-box path can keep the old size. The replacement uses an explicit shared canvas and no implicit expansion. Tests verify known coordinates and pixel/box alignment for rotation and shear, including odd dimensions and empty targets.

After rotation/shear, labels remain **axis-aligned enclosing boxes** because YOLOS uses axis-aligned detection. These can include more background than the unrotated box. Area is recomputed before YOLOS target conversion. Final 512x512 letterboxing transforms all boxes again; the processor does not resize a second time.

Uniform scaling would otherwise be cancelled by fitting back into a fixed canvas. The scaling recipes therefore use a common reference canvas 1.25 times the source dimensions. The 1.15x content is larger than the 0.8x content within that reference; it is not a crop-based zoom and need not be larger than the unpadded original view. Width/height stretch recipes independently change the image aspect ratio.

All transforms preserve class IDs. This implements the user's confirmed interpretation of expansion/collapse as geometric scaling, not a change in defect type. Stronger transformations can still reduce realism or obscure small features; inspect the generated grids before judging an experiment.

## Counts and training behavior

There are still **63 independent source training images**, with **1,260 presentations per epoch**:

| Training image class | Sources | Presentations |
|---|---:|---:|
| bridge | 14 | 280 |
| gap | 3 | 60 |
| line_collapse | 14 | 280 |
| multi_bridge | 6 | 120 |
| no_defect | 26 | 520 |
| Total | 63 | 1,260 |

These are generated on-the-fly, not saved as independent training data. Raw files, annotations and source splits remain unchanged. Validation/test remain 16 deterministic images each. Clean images keep empty targets for every transform.

Use **natural sampling** to guarantee that all 20 slots for every image are included. Weighted sampling is supported, but draws 1,260 views with replacement and cannot guarantee coverage of every source/recipe. Its frequencies are computed from the 63 source images, not the 1,260 views.

Training-view mode, recipes and exposure counts are saved in run_config.json and sampling_report.json. Epoch metrics record training presentations and optimizer-update opportunities. At microbatch 1 and accumulation 8 there are 158 update opportunities per epoch, compared with 8 for the 63-view baseline. The 15/40 epoch limits are unchanged, so this is substantially more training work; no claim of equal-compute comparison is made. This also does not change class imbalance ratios or guarantee improved AP.

## Commands (PowerShell, from the project folder)

```powershell
$py = '.\venv\Scripts\python.exe'

# Preview one representative per class plus clean, with all 20 views and grids:
& $py .\scripts\preview_data.py --config .\configs\pipeline_config_20_views.json --output .\outputs\checks\20_views_preview_new

# Optional: export inspection previews for ALL 63 training images and every view:
& $py .\scripts\preview_data.py --config .\configs\pipeline_config_20_views.json --all-images --output .\outputs\checks\20_views_all_training

# Focused checks:
& $py -m unittest discover -s tests -v

# Two-step smoke check on transformed training images, not full training:
& $py .\scripts\smoke_test.py --config .\configs\pipeline_config_20_views.json --size 512 --output .\outputs\checks\smoke_20_views_new

# Train a fresh experiment, showing every source/recipe each epoch:
& $py .\scripts\train_yolos.py --config .\configs\pipeline_config_20_views.json --sampling natural --output .\outputs\runs\natural_20_views_01 --device auto

# Select a threshold and measure validation performance:
& $py .\scripts\evaluate.py --model .\outputs\runs\natural_20_views_01\best --split val --select-threshold --output .\outputs\runs\natural_20_views_01\validation

# Readable epoch summary and plots:
& $py .\scripts\summarize_run.py --run .\outputs\runs\natural_20_views_01
```

Use fresh output names. Do not resume an older experiment with the changed view configuration. To use 15 total views, copy the config and retain 14 entries in augmentation_recipes; the original view is added automatically. Existing one-view and two-view configurations remain supported.

## Verification artifacts

- `outputs/checks/20_view_tests.txt`: 23 passing tests, including all 1,260 training views at a 128px test canvas, all source boxes retained, known rotation/stretch/shear coordinates, full-frame preservation and grayscale noise sigma.
- `outputs/checks/20_view_previews_verified/`: 100 annotated 512px previews plus five contact sheets, covering four defect classes and clean images. Gap and multi_bridge grids were visually inspected.
- `outputs/checks/smoke_20_views_verified/report.json`: real 512px pretrained-model forward/backward and optimizer-step check on rotated empty and combined shear/rotation multi-object views; validation-metric smoke check on development data. See the saved report for completion and loss values.

No full training or test-set evaluation is launched by these implementation checks. Original failed previews/check outputs are not used as training inputs.

## Stronger weighted sampling experiment

`configs/pipeline_config_20_views_weighted_balanced.json` keeps the same 20 augmentation views and training budgets, but sets `sampling_power=1.0` and `sampling_cap=5.0`. Run it with `--sampling weighted`. Training-image counts are bridge 14, gap 3, line_collapse 14, multi_bridge 6, clean 26. The resulting individual weights are about 1.0, 4.67, 1.0, 2.33 and 1.0, respectively. The sampler makes 1,260 draws with replacement per epoch. Expected class exposure is approximately 216 presentations for each defect class and 400 clean presentations, assuming single-class training images. Actual draws vary by epoch; views may repeat or be omitted. Repeated views do not add independent source images.

Use a fresh run directory and calibrate its best checkpoint on validation before one final test evaluation.
