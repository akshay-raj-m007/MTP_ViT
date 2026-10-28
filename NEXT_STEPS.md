# Next steps: verification, risks, experiments

Checked against PROJECT_SUMMARY.md, the code and the small result JSONs, with no images, weights or logs opened. `$py = .\venv\Scripts\python.exe`, `$ssd = .\outputs\mobilevitv2_ssd\env\Scripts\python.exe`.

## 1. Verification: mismatches only

**Verified and matching:**
- Split image, box and clean counts, per-class box and image counts, and the gap filenames.
- 19 recipes plus the original (20 views, 1,260 presentations per epoch).
- Weights 4.67 and 2.33 (power 1, cap 5).
- SSD expected gap presentations of 5.86 per epoch.
- Thresholds 0.55 and 0.60, and best epochs B9, B13, B38 and B38.
- Every validation and test metric table, including P/R/F1, TP/FP/FN, 0/3 gap recall and 100% clean specificity.
- SSD localization recall 4/13 and 5/13.
- 31 near-duplicate candidates, 13 of them crossing splits, and a smallest image of 81×81.
- All SSD postprocessing values.

| # | Summary says | Code/results show |
|---|---|---|
| M1 | YOLOS Stage B "up to 40 epochs" (§7). §13.10 lists the confounds. | Early stopping (`patience=8`, [train_yolos.py:90](scripts/train_yolos.py#L90)) ended Stage B at 17 epochs (natural) and 21 (weighted). SSD has `early_stopping=false` and ran all 40. This confound is missing from §13.10. |
| M2 | Only SSD is described as FP32. | YOLOS trained with bf16 autocast ([train_yolos.py:66](scripts/train_yolos.py#L66), `outputs/runs/*/device.json`). This is another confound. |
| M3 | "Padded scaling at 0.8× and 1.15×" (§6). | Because of the 1.25× reference canvas ([multi_augment.py:226-231](sem_yolos/multi_augment.py#L226-L231)), the defect ends up at **0.64×** and **0.92×** of the original view. Neither recipe enlarges anything. |
| M4 | 13 cross-split near-duplicates make val/test independence uncertain (§13.6). | **All 13 crossing pairs are zero-box clean images** (figs 53–78; 1 val and 4 test images involved). No defect image crosses a split. The 4 candidates that involve defect images (fig56/92/95 and fig81/82) are all within one split. The leakage mainly inflates clean specificity and FP/clean-image, not defect recall. Crop-level leakage cannot be detected by the 9×8 dHash. |
| M5 | Splits are preserved unless `--resplit` is used (§3). | Correct, but the current gap split is a manual manifest edit (commit `ede0629`). `--resplit` would replace it with the quota split ([prepare_data.py:122-150](scripts/prepare_data.py#L122-L150)). Whether it would reproduce the same gap assignment is UNVERIFIED (not run). |
| M6 | n/a (documentation) | `run.txt` and the SSD README name `natural_20_views_01` and `natural_seed42`. The actual runs are `unweighted_gap_resplit_01`, `weighted_gap_resplit_01` and `natural_seed42_retry_20261027_230508`. |

**New fact, not in the summary:** at 512 px, 4 of the 6 **training** gap boxes have a short side of 11–13 px, smaller than one 16-px YOLOS patch. Validation gaps are 19–30 px, and test gaps are 15, 15 and 112 px. At SSD's 320 resize, the training gaps are about 7–8 px. So gaps are both sub-patch and scale-mismatched between splits.

## 2. Bugs and methodology risks

| # | Sev | Location | Issue | One-line fix |
|---|---|---|---|---|
| R1 | High | [prepare_data.py:173-183](scripts/prepare_data.py#L173-L183), `split_manifest.json` | The gap split was hand-edited after earlier results, and nothing can regenerate it. The test set may have been adapted to. | Commit an explicit split-override/groups file, and label the current test results as post-hoc. |
| R2 | High | `size` in the configs, [preprocessing.py:24-27](experiments/mobilevitv2_ssd/sem_ssd/preprocessing.py#L24-L27) | Training gaps are 11–13 px at 512, below the 16-px patch, and about 7 px at SSD 320. The model may not be able to represent them. | Test with higher input size or gap-centred tiles (E3). |
| R3 | Med | [train_yolos.py:105-115](scripts/train_yolos.py#L105-L115), [evaluate.py:171-173](scripts/evaluate.py#L171-L173), [training.py:207-227](experiments/mobilevitv2_ssd/sem_ssd/training.py#L207-L227) | The checkpoint and the threshold are both selected on the same 16 val images, so the validation tables are optimistically biased. | Label val metrics as "selection". Use grouped CV for model ranking. |
| R4 | Med | [prepare_data.py:56-57](scripts/prepare_data.py#L56-L57), [prepare_data.py:76-79](scripts/prepare_data.py#L76-L79) | Near-duplicate candidates are only reported, never grouped. The dHash is whole-image and coarse. | Union the candidate components through `--groups` and add a patch-level similarity check. |
| R5 | Med | [runtime.py:192-193](sem_yolos/runtime.py#L192-L193) vs `model.py:177-180`, [data.py:14-24](sem_yolos/data.py#L14-L24) vs [preprocessing.py:24-27](experiments/mobilevitv2_ssd/sem_ssd/preprocessing.py#L24-L27) | Further confounds: YOLOS produces one argmax class per query (100 per image), while SSD class-aware NMS can produce several class hypotheses per location (up to 200; COCOeval keeps 100). Letterbox/bilinear vs stretch/bicubic. Plus M1 and M2. | Run E2, a matched-pipeline comparison. |
| R6 | Med (UNVERIFIED) | `model.py:126-130` | The upstream multibox loss mines negatives in proportion to positives, so a clean image likely gives no negatives. Only the 3 hardest anchors teach background. | Sweep `empty_negatives` (for example 3 vs 50). Confirm the upstream `SSDLoss` behaviour when `num_pos=0`. |
| R7 | Low | [multi_augment.py:224-231](sem_yolos/multi_augment.py#L224-L231) | The "scale_up" recipe actually shrinks (M3). Small gaps never get enlarged views. | For zoom > 1 use `canvas = max(h, nh)` and allow a defect-safe crop, or rename the recipe. |
| R8 | Low | [multi_augment.py:251-256](sem_yolos/multi_augment.py#L251-L256), [data.py:88-89](sem_yolos/data.py#L88-L89) | Noise is added after letterboxing, so the grey padding gets noisy too. That is a cue that only appears in augmented views. | Mask out the padding before adding noise, or add noise before the letterbox. |
| R9 | Low | [multi_augment.py:196-207](sem_yolos/multi_augment.py#L196-L207) | Rotated and sheared boxes are correct (enclosing box of the 4 transformed corners) but get inflated. ±12° on thin boxes adds a lot of background to the IoU target. | Keep. Optionally use ≤5° for the gap and bridge images. |
| R10 | Low | [metrics.py:153-158](sem_yolos/metrics.py#L153-L158) | The threshold objective is pooled detection F1, so gap has essentially no influence on it. | Also report the per-class or macro-F1 objective. |
| R11 | Low | [prepare_data.py:210-211](scripts/prepare_data.py#L210-L211) | It always overwrites `docs/DATASET_REPORT.md`, even when `--output` points elsewhere. | Write the report into the output directory. |

**Checked with no issue found:**
- Weighted sampler: weights are repeated per view in the same order as `view_info` ([data.py:121](sem_yolos/data.py#L121)).
- Natural mode shuffles every source/recipe combination once.
- SSD labels: +1 at [dataset.py:225](experiments/mobilevitv2_ssd/sem_ssd/dataset.py#L225) and −1 at [training.py:74](experiments/mobilevitv2_ssd/sem_ssd/training.py#L74).
- YOLOS labels: 0–3, with no-object as the last logit.
- COCOeval: positive local IDs, `iscrowd=0`, area "all", maxDets 100.
- No exact duplicates across splits, and SSD checks hashes per split.
- UNVERIFIED: whether the CVNets checkpoint expects RGB in [0, 1] (`/255`, [preprocessing.py:40](experiments/mobilevitv2_ssd/sem_ssd/preprocessing.py#L40)).

## 3. Next 5 experiments (ranked by information per GPU-hour)

Cost unit **R** is one YOLOS 20-view run at 512 (about 40k presentations). Absolute hours are UNVERIFIED because the run-file timestamps were copied, not recorded (see Q1).

**E1. Train-split gap probe** (CPU, about 0 R; under 30 min)
- **Question:** can the trained models detect gaps on their own training images? If not, the cause is representation or labels; if yes, it is generalization.
- **Edit:** at [evaluate.py:155](scripts/evaluate.py#L155), change the choices to `['train','val','test']`.
- **Command:** `& $py .\scripts\evaluate.py --model .\outputs\runs\unweighted_gap_resplit_01\best --split train --output .\outputs\runs\unweighted_gap_resplit_01\train_probe --device cpu`. Repeat for the weighted run.
- **SSD:** add a `--split` argument used at `experiment.py:79`. `SemDataset` already allows `train`.
- **Compare:** train gap AP50 and gap recall at the saved threshold, against the train AP for the other classes.
- **Also:** fill in `manual_review.csv` (human time, no GPU).

**E2. Matched-pipeline YOLOS vs SSD** (CPU or small GPU, about 0.05 R)
- **Question:** how much of the 17-point validation mAP gap comes from the architecture rather than the rest of the pipeline?
- **Config:** copy `configs/pipeline_config.json` to `pipeline_config_ssd_matched.json` with `size: 320`, `training_view_mode: "random_augmented"` (the same flips and jitter as SSD, 63 presentations per epoch) and `patience: 99` (no early stopping).
- **Command:** `& $py .\scripts\train_yolos.py --config .\configs\pipeline_config_ssd_matched.json --sampling natural --output .\outputs\runs\yolos_ssd_matched_01 --device cpu`. `--device cpu` also removes the bf16 confound.
- **Optional SSD arm:** a second run with `resize_mode: "letterbox"`.
- **Compare:** validation mAP and per-class AP against SSD natural (8.68%) and YOLOS natural (25.65%).

**E3. Gap resolution test** (small GPU with ≥6 GB is likely enough at batch 1; about 3 R; UNVERIFIED memory)
- **Question:** is the sub-patch scale (R2) what blocks gap detection?
- **Config:** copy `pipeline_config_20_views.json` with `size: 1024`, so training gaps become about 22–26 px, or `768` for about 1.6 R.
- **Command:** `& $py .\scripts\train_yolos.py --config .\configs\pipeline_config_20_views_1024.json --sampling natural --output .\outputs\runs\natural_1024_01`, then evaluate with `--select-threshold` on val.
- **Compare:** validation gap AP50 and gap recall at the threshold, plus validation mAP. Rerun E1 on this model.
- **Gate:** run E3 only if E1 shows near-zero gap detection on the training images.

**E4. Seed replicates for the sampling conclusion** (GPU, 4 R)
- **Question:** is the weighted vs natural difference (about ±3 mAP) larger than seed noise?
- **Edit:** copy both configs with `seed: 43` and `seed: 44`, then train and evaluate as in [CODE_MAP.md](CODE_MAP.md).
- **Compare:** mean ± SD of validation and test mAP and per-class AP. Report a difference only if it exceeds 2 SD.

**E5. Leakage sensitivity, then a grouped split** (CPU, 0 R for step 1)
- **Step 1:** re-score the saved `outputs/runs/*/{validation,final_test}/predictions.json` and the SSD `best_validation_predictions.json` with `sem_yolos.metrics.evaluate_predictions`. Drop the 5 cross-linked clean images: val `aa0c08c7-fig65`, and test `4befb5e7-fig53`, `073a7321-fig64`, `5afa7ab5-fig67`, `04083246-fig71`. Compare mAP and FP/clean-image.
- **Step 2:** write `groups.json` from the 31-pair components and run `& $py .\scripts\prepare_data.py --groups groups.json --resplit --output data/prepared/grouped_v1`. Back up `docs/DATASET_REPORT.md` first (R11). Rerun the best pipeline on that split (1 R per model).

## 4. Questions for you

1. Which GPU did you use, and how long did one YOLOS 20-view run and one SSD run take? This turns the R units into hours.
2. Was the gap split revision informed by looking at test-set predictions or metrics, or only by inspecting the annotations? This decides whether the current test set can still be reported as held-out.
3. Can you confirm the source groups for figs 53–78, fig56/92/95 and fig81/82 (same paper, same figure)?
4. Is a 768–1024 input or gap-centred tiling acceptable for your deployment target?
5. Do you have, or can you annotate, any additional gap images, even from one new source? E1–E3 can diagnose the gap failure but cannot add new gap examples.
