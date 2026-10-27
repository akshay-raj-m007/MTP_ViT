# Current implementation state

Updated 2026-10-27. Continue in this experiment directory; do not recreate it. No full training or final test evaluation has been run.

Implemented: official complete COCO SSD–MobileViTv2-0.75 strict loading and checksum verification; four defect classes plus internal SSD background; read-only reviewed data adapter; joint image/box transforms; original and transformed previews; natural and mild weighted sampling with equal presentation budgets; two-stage freezing and BatchNorm control; empty-image hard background loss; accumulation including partial groups; CUDA precision controls; epoch-boundary resume; corrected whole-validation COCO AP and fixed-threshold class-agnostic localization recall; checkpoint/prediction/provenance saving. Commands are in README.md.

Verified before the continuation additions: 10 focused tests passed, 320 and 512 CPU forwards passed, complete checkpoint loaded with no missing/unexpected keys, three smoke optimizer steps (gap, empty, multi-object) and one validation-image decode passed. Frozen encoder BatchNorm remained unchanged. Three training gap images were included in 14 transformed previews. Dataset structural checks passed: 63/16/16 train/val/test images, 44/13/12 boxes. Test images were inspected structurally only, never evaluated for performance.

The final continuation adds class-agnostic localization recall at confidence 0.3 / IoU 0.5 (diagnostic only, not used for checkpoint selection), original-size annotation preview pairs, and an additional focused test. Final verification: **11 tests passed in 49.87 seconds**. The updated preview command generated seven original-size pairs and 14 transformed pairs in `outputs/mobilevitv2_ssd/preview_20261027_223918_528716/`; a gap original/transformed example was visually inspected. Git confirmed this output directory is ignored. Both staged and unstaged tracked-file diffs were empty. No TODO/FIXME/NotImplemented markers remain in the new experiment.

All 271 pre-existing tracked files matched the original SHA256 snapshot; `git diff --stat` was empty. Only the new experiment is untracked. No commit, push, deletion, split regeneration, old environment modification or existing YOLOS code/result change was performed. Generated outputs, upstream checkout and separate environment are ignored under `outputs/mobilevitv2_ssd/`. The hash audit is `outputs/mobilevitv2_ssd/isolation_audit.json`.

No known CPU setup blocker. GPU/CUDA execution, Linux execution, full training, tiny-subset overfitting and end-to-end interrupted-training resume remain unverified. Use the GPU smoke check before committing to training. No model accuracy claim follows from smoke checks.

Next command on a fresh Windows GPU machine, from the repository root with Python 3.12 installed:

```powershell
py -3.12 -m venv .\outputs\mobilevitv2_ssd\env
```

Then follow README.md setup/install/inspect/preview/test/smoke commands before its explicit natural and weighted training commands. If the separate CUDA environment and upstream checkpoint are already set up, the next verification command is:

```powershell
& .\outputs\mobilevitv2_ssd\env\Scripts\python.exe .\experiments\mobilevitv2_ssd\scripts\experiment.py smoke --device cuda --precision fp32
```

The locally created environment is CPU-only; it is not a substitute for the CUDA environment on the GPU machine. Do not transfer virtual environment directories between operating systems.
