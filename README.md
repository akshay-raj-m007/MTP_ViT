# YOLOS SEM defect detection

## Project layout

```text
configs/                    Training configuration JSON files
scripts/                    Training, evaluation, inference and preparation commands
sem_yolos/                  Shared dataset, augmentation, model and metrics code
data/
  raw/images/               Original 95 images
  raw/result.json           Original COCO annotations
  prepared/label_studio/    train.json, val.json, test.json and split manifest
outputs/
  runs/                     Training checkpoints, metrics and reports
  checks/                   Augmentation previews and verification artifacts
docs/                       Usage, augmentation details and dataset reports
archive/                    Older Roboflow datasets, models, results and scripts
tests/                      Existing automated tests
venv/                       Existing Python environment
requirements.txt            Python dependencies
```

The existing split is **63 training / 16 validation / 16 test images**. Folder relocation preserves the assignments and dataset fingerprint. Existing runs are now under `outputs/runs/`; their saved configurations remain intact.

## Train with one original and 19 augmented views per image

Run these commands in PowerShell:

```powershell
Set-Location 'D:\aksha\MTP\annotate_image.v4-train-test.coco'
$py = '.\venv\Scripts\python.exe'

& $py .\scripts\train_yolos.py `
    --config .\configs\pipeline_config_20_views.json `
    --sampling natural `
    --output .\outputs\runs\natural_20_views_01 `
    --device auto
```

Use a new output directory for each experiment. Natural sampling visits all **1,260 views per epoch**. Augmentations are generated on the fly with corresponding bounding-box transforms; preview files are for inspection. Validation and test use the original images with deterministic preprocessing.

For stronger rare-class promotion, use `configs/pipeline_config_20_views_weighted_balanced.json` with `--sampling weighted` and a separate output such as `outputs/runs/weighted_20_views_balanced_01`. This changes only the sampler weight power and cap relative to the 20-view config. Training still makes 1,260 draws per epoch. Weighted sampling uses replacement, so it does not guarantee every original or recipe view appears in every epoch.

## Preview augmentation before training

```powershell
& $py .\scripts\preview_data.py `
    --config .\configs\pipeline_config_20_views.json `
    --output .\outputs\checks\20_views_preview_new
```

## Evaluate after training

```powershell
& $py .\scripts\evaluate.py `
    --model .\outputs\runs\natural_20_views_01\best `
    --split val `
    --select-threshold `
    --output .\outputs\runs\natural_20_views_01\validation

& $py .\scripts\summarize_run.py --run .\outputs\runs\natural_20_views_01
```

Threshold selection is performed once on validation data. Reserve test evaluation until experiment choices are final. An existing baseline can be accessed at `outputs/runs/natural_01/best`.

## Documentation

- [GitHub and NVIDIA GPU setup](docs/GPU_GITHUB_SETUP.md)
- [Detailed usage](docs/USAGE.md)
- [Twenty-view augmentation recipes](docs/AUGMENTATION_20_VIEWS.md)
- [Folder migration and path compatibility](docs/PROJECT_STRUCTURE.md)
- [Dataset report](docs/DATASET_REPORT.md)
- [COCO annotation ID 0 metric correction](docs/COCO_ANNOTATION_ID_0_FIX.md)
- [Implementation notes](docs/IMPLEMENTATION_NOTES.md)
