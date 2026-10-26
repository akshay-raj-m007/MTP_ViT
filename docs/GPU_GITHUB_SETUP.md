# Clone and train on an NVIDIA GPU

This guide uses PowerShell on the new Windows machine. Paths in the code and prepared annotations are relative to the cloned repository, so the clone may live in any directory. The checked-in dataset contains 95 raw images, `result.json`, and the fixed 63/16/16 train/validation/test split. The 20-view augmentation is generated in memory each training epoch; there are no augmented files to transfer.

## 1. Put the project on GitHub

Create an **empty private** repository in your GitHub account. From the current project folder, run:

```powershell
Set-Location 'D:\aksha\MTP\annotate_image.v4-train-test.coco'
git init
git add .
git status --short
git commit -m "Prepare YOLOS project and dataset for GPU training"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Check `git status --short` before committing: it should show `data/raw/images/`, `data/raw/result.json`, prepared split files, code, configs and docs. It should not show `venv/`, `outputs/`, `archive/`, or `run.txt`, which `.gitignore` excludes. The 95 raw images total about 10 MiB, and the largest is about 1.33 MiB, so Git LFS is not needed for these images. Existing run checkpoints are deliberately local; copy one separately only if you need its weights on the new machine.

If GitHub already has a repository you want to use, use its remote URL instead of creating a new repository. If `origin` exists, inspect `git remote -v` before changing it.

## 2. Clone and create a fresh environment

On the GPU machine, install Git, an NVIDIA driver and a Python version supported by your selected PyTorch CUDA build. The dependency pins were verified with Python 3.14.2 on the source machine. Then:

```powershell
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
Set-Location .\YOUR_REPOSITORY
py -3.14 -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install --upgrade pip
nvidia-smi
```

For the same PyTorch and torchvision versions used in development, install their official CUDA 12.6 wheels ([PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)):

```powershell
& $py -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
```

The NVIDIA driver must support the selected CUDA runtime. If the new machine needs another supported build, choose a matched pair from the [official PyTorch installer](https://pytorch.org/get-started/locally/). Install the remaining pinned project dependencies:

```powershell
& $py -m pip install -r requirements-gpu.txt
& $py -c "import torch, torchvision; print('torch:', torch.__version__, 'CUDA build:', torch.version.cuda, 'GPU available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Proceed with `--device cuda` only when the command reports `GPU available: True`. If it says false, check the driver, selected CUDA wheel, and whether the commands used the same Python environment. An RTX 3050 can have different VRAM capacities; check `nvidia-smi` rather than assuming a memory size.

## 3. Confirm dataset and augmentation

```powershell
& $py .\scripts\prepare_data.py
& $py .\scripts\preview_data.py `
    --config .\configs\pipeline_config_20_views_weighted_balanced.json `
    --allow-download `
    --output .\outputs\checks\weighted_preview_01
```

Preparation reuses the committed split assignments and checks the raw images and annotations. It must report 63 training, 16 validation and 16 test images. The preview fetches the YOLOS image processor on first use and writes a few example grids for visual inspection. Use `--all-images` only if you want all training previews; it is not required for training. The actual augmentation is computed by `sem_yolos/multi_augment.py` and `sem_yolos/data.py` during each training batch, with transformed bounding boxes.

## 4. Weighted training and evaluation

```powershell
& $py .\scripts\train_yolos.py `
    --config .\configs\pipeline_config_20_views_weighted_balanced.json `
    --sampling weighted `
    --output .\outputs\runs\weighted_20_views_balanced_01 `
    --device cuda `
    --allow-download

& $py .\scripts\evaluate.py `
    --model .\outputs\runs\weighted_20_views_balanced_01\best `
    --split val `
    --select-threshold `
    --output .\outputs\runs\weighted_20_views_balanced_01\validation `
    --device cuda

& $py .\scripts\evaluate.py `
    --model .\outputs\runs\weighted_20_views_balanced_01\best `
    --split test `
    --output .\outputs\runs\weighted_20_views_balanced_01\final_test `
    --device cuda
```

`--allow-download` downloads the pretrained `hustvl/yolos-tiny` weights the first time; subsequent use of the saved checkpoint is local. The run's `sampling_report.json` records the weighting and 20 views. Validation chooses a threshold for this checkpoint, which the test command uses unchanged. Run test after choosing the experiment on validation.

The configuration starts at a 512-pixel canvas and batch size 1. If CUDA runs out of memory, make a **copy** of the JSON configuration and lower `size` to 384 or 256 (a multiple of 16), then start a new output directory. This changes the experiment, so keep the revised config with its results. Do not resume a run with changed canvas size.

To continue an interrupted run after a completed epoch, repeat the training command with the same config/output and add `--resume`. The run's `last/` checkpoint contains the resume state. Keep `outputs/runs/` backed up separately; `.gitignore` keeps large checkpoints out of source Git history.

On Linux, use `python3 -m venv .venv`, `./.venv/bin/python`, and the Linux CUDA wheel from the PyTorch previous-versions page. The repository-relative data paths are portable.
