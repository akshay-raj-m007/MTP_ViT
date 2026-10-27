# Isolated SSD–MobileViTv2 SEM experiment

This experiment predicts bounding boxes and per-box classes: bridge, gap, line_collapse and multi_bridge. External IDs are 0–3; SSD uses background 0 and defects 1–4. Clean images have zero boxes. Whole-image classification, threshold selection, segmentation and final test evaluation are deferred.

Everything here is new. The YOLOS environment, source, data, split files and results are used read-only. Generated artifacts belong in `outputs/mobilevitv2_ssd/`, which the existing repository ignore rules cover. Do not commit that environment or the downloaded upstream checkout. No changes to the repository's `.gitignore` are needed.

## Setup on Windows PowerShell

Run from the repository root. Python 3.12 and Git must be available. The first command on a fresh GPU machine is:

```powershell
py -3.12 -m venv .\outputs\mobilevitv2_ssd\env
```

Optional activation: `& .\outputs\mobilevitv2_ssd\env\Scripts\Activate.ps1`. The commands below use the environment's executable directly, so activation is not required.

Then install the separate environment. The CUDA 12.1 wheel is a [published PyTorch 2.5.1 installation option](https://pytorch.org/get-started/previous-versions/#v251), paired with torchvision 0.20.1. An NVIDIA driver supporting CUDA 12.1 is required; a separately installed CUDA toolkit is unnecessary for these wheels. Check the driver with `nvidia-smi`.

```powershell
$ssdPython = '.\outputs\mobilevitv2_ssd\env\Scripts\python.exe'
$ssdEntry = '.\experiments\mobilevitv2_ssd\scripts\experiment.py'
$natural = '.\experiments\mobilevitv2_ssd\configs\natural.json'
$weighted = '.\experiments\mobilevitv2_ssd\configs\weighted.json'
& $ssdPython -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
& $ssdPython -m pip install -r .\experiments\mobilevitv2_ssd\requirements.txt
& $ssdPython $ssdEntry setup
& $ssdPython $ssdEntry check-setup --device cuda --precision fp32
```

For CPU-only checks, substitute `https://download.pytorch.org/whl/cpu` in the torch installation and use `--device cpu`. This workspace already has a separate CPU environment created at that location. Do not copy virtual environments between machines; recreate them.

`setup` downloads the pinned official CVNets source and complete COCO checkpoint and verifies its SHA256. To supply an already-downloaded checkpoint, run `setup --skip-weights`, then pass `--checkpoint 'D:\path\coco-ssd-mobilevitv2-0.75.pt'` to `check-setup`, `smoke` and fresh `train`/`overfit` commands. It must match the verified official checksum. Setup requires internet; subsequent training can be offline.

## Inspect, preview and verify

```powershell
& $ssdPython $ssdEntry inspect --config $natural
& $ssdPython $ssdEntry inspect --config $weighted
& $ssdPython $ssdEntry preview --config $natural
& $ssdPython -m pytest .\experiments\mobilevitv2_ssd\tests -q -p no:cacheprovider
& $ssdPython $ssdEntry smoke --device cuda --precision fp32
# Optional supported-resolution check; does not train.
& $ssdPython $ssdEntry check-setup --size 512 --device cuda --precision fp32
```

Each inspection/preview/smoke invocation makes a fresh timestamped output directory and prints its path. Previews include all training gap images, one clean image and the other classes; each shows the input beside its box overlay. Augmentation is performed on the fly and boxes flip with images. These previews are not an expanded training dataset.

The preview folder also contains `*_original.png` pairs at the original image dimensions for direct annotation review. Validation reports include class-agnostic localization recall at fixed confidence 0.3 and IoU 0.5, using score-ordered one-to-one matching after the detector's class-aware NMS. This diagnostic ignores predicted classes; checkpoint selection still uses class-aware COCO mAP only. The threshold is not tuned per epoch.

Optional diagnostic overfit check, explicitly separate from real experiments:

```powershell
& $ssdPython $ssdEntry overfit --config $natural --subset-size 4 --epochs 30 --device cuda --precision fp32 --output outputs/mobilevitv2_ssd/overfit_seed42
```

This uses four deterministic training images, including gap and clean examples when available, no augmentation, accumulation 1, Stage A 5 epochs and Stage B 30 epochs. Its AP is measured on that same training subset and cannot estimate generalization. This command was not run during implementation.

## Paired training, validation and resume

Both runs use the same model, seed, precision, augmentation, validation schedule and number of presentations. The configurations differ only in sampling. These commands start training only when you run them:

```powershell
& $ssdPython $ssdEntry train --config $natural --seed 42 --device cuda --precision fp32 --output outputs/mobilevitv2_ssd/natural_seed42
& $ssdPython $ssdEntry train --config $weighted --seed 42 --device cuda --precision fp32 --output outputs/mobilevitv2_ssd/weighted_seed42
```

Fresh runs refuse an existing output directory. To resume an interrupted run from its last completed epoch:

```powershell
& $ssdPython $ssdEntry train --config $natural --seed 42 --device cuda --precision fp32 --output outputs/mobilevitv2_ssd/natural_seed42 --resume
& $ssdPython $ssdEntry train --config $weighted --seed 42 --device cuda --precision fp32 --output outputs/mobilevitv2_ssd/weighted_seed42 --resume
```

Resume checks the dataset and complete configuration fingerprints, upstream revision, precision, device type and subset setting. It restores model, optimizer, scheduler, scaler, stage, completed epoch, counters, Python/NumPy/torch/CUDA RNGs and separate sampling/augmentation/loader RNGs. An interrupted partial epoch is replayed from the previous saved epoch. GPU kernels can still introduce nondeterminism; bitwise equivalence across machines is not promised.

Validation is already run after every epoch. To regenerate selected-checkpoint predictions later:

```powershell
& $ssdPython $ssdEntry validate --model outputs/mobilevitv2_ssd/natural_seed42/best.pt --device cuda --precision fp32
& $ssdPython $ssdEntry validate --model outputs/mobilevitv2_ssd/weighted_seed42/best.pt --device cuda --precision fp32
```

For all three paired seeds, use fresh names below instead of the individual seed42 training commands:

```powershell
foreach ($seed in 42,43,44) {
    & $ssdPython $ssdEntry train --config $natural --seed $seed --device cuda --precision fp32 --output "outputs/mobilevitv2_ssd/paired_natural_seed$seed"
    if ($LASTEXITCODE -ne 0) { throw 'Natural run failed' }
    & $ssdPython $ssdEntry train --config $weighted --seed $seed --device cuda --precision fp32 --output "outputs/mobilevitv2_ssd/paired_weighted_seed$seed"
    if ($LASTEXITCODE -ne 0) { throw 'Weighted run failed' }
}
```

Default precision is explicit fp32 in these examples. `fp16`, `bf16` and `auto` are supported on CUDA, with capability checks and GradScaler only for fp16. If changing precision or resolution, do so identically for both new runs. For 512 pixels add `--size 512` to both training commands and the matching resume commands. CPU forward shapes were verified at 320 and 512; GPU memory has not been measured. Initial default 320 matches the downloaded detector's official evaluation configuration and reduces memory demand on an RTX 3050.

## Linux / WSL2

Windows CPU imports, strict loading and smoke checks succeeded; there is no identified Windows-specific blocker. Linux commands use the same Python files and relative dataset paths, without sudo or `py`:

```bash
python3.12 -m venv outputs/mobilevitv2_ssd/env
ssd_python=outputs/mobilevitv2_ssd/env/bin/python
ssd_entry=experiments/mobilevitv2_ssd/scripts/experiment.py
natural=experiments/mobilevitv2_ssd/configs/natural.json
weighted=experiments/mobilevitv2_ssd/configs/weighted.json
"$ssd_python" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
"$ssd_python" -m pip install -r experiments/mobilevitv2_ssd/requirements.txt
"$ssd_python" "$ssd_entry" setup
"$ssd_python" "$ssd_entry" check-setup --device cuda --precision fp32
"$ssd_python" "$ssd_entry" inspect --config "$natural"
"$ssd_python" "$ssd_entry" inspect --config "$weighted"
"$ssd_python" "$ssd_entry" preview
"$ssd_python" -m pytest experiments/mobilevitv2_ssd/tests -q -p no:cacheprovider
"$ssd_python" "$ssd_entry" smoke --device cuda --precision fp32
# Optional diagnostic; not part of the paired runs:
"$ssd_python" "$ssd_entry" overfit --subset-size 4 --epochs 30 --device cuda --precision fp32 --output outputs/mobilevitv2_ssd/overfit_seed42
for seed in 42 43 44; do
  "$ssd_python" "$ssd_entry" train --config "$natural" --seed "$seed" --device cuda --precision fp32 --output "outputs/mobilevitv2_ssd/natural_seed$seed" || break
  "$ssd_python" "$ssd_entry" train --config "$weighted" --seed "$seed" --device cuda --precision fp32 --output "outputs/mobilevitv2_ssd/weighted_seed$seed" || break
done
# Resume the relevant interrupted run:
"$ssd_python" "$ssd_entry" train --config "$weighted" --seed 42 --device cuda --precision fp32 --output outputs/mobilevitv2_ssd/weighted_seed42 --resume
"$ssd_python" "$ssd_entry" validate --model outputs/mobilevitv2_ssd/weighted_seed42/best.pt --device cuda --precision fp32
```

If `python3` is already Python 3.12, it can replace `python3.12`. Do not run the old `scripts/prepare_data.py` for this experiment; the reviewed prepared files are reused as supplied.

## Outputs and research interpretation

Each training directory contains `run_config.json`, `dataset_report.json`, `sampling_report.json`, `runtime.json`, `pretrained_load.json`, explicit trainable-module lists for each stage, per-epoch `metrics_A_*.json` / `metrics_B_*.json`, `stage_A_best.pt`, `stage_B_best.pt`, `best.pt`, and resumable `last.pt`. The best checkpoint also has validation metrics, canonical COCO predictions and selection provenance saved alongside it. Epoch reports include losses, AP, presentations, actual optimizer updates, update opportunities, duration and peak CUDA allocated memory.

Select natural versus weighted using validation and report variation over seeds. Do not select using test results. Whole-image classification and a confidence threshold can later be derived from validation detections, then fixed before final test evaluation. There is intentionally no test-evaluation command here.

Default training has 63 presentations and 8 optimizer-update opportunities per epoch: 15 Stage A + 40 Stage B epochs = 3,465 presentations and 440 update opportunities. AMP can skip unsafe updates; actual counts are recorded. Early stopping is disabled for a controlled budget; `--early-stopping` enables patience 8 for exploratory runs and changes that comparison. Stage B loads the best Stage A weights, not its final epoch. `best.pt` is the best validation mAP checkpoint across both stages.

This conservative single-view setup differs from historical YOLOS's 20 presentations per source image, augmentation recipes, preprocessing and update budgets. A comparison with those runs is a **pipeline comparison**, not an architecture-only ablation. Improved gap performance or overall accuracy is a hypothesis to test, not a promised outcome.

See [RESEARCH_NOTES.md](RESEARCH_NOTES.md) for exact architecture, license, preprocessing, loss adaptation, dataset and verification details.
