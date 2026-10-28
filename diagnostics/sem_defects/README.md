# Read-only SEM gap diagnostics

## Ready for manual review (2026-10-28)

The completed local combined report is `outputs/diagnostics_sem/combined_review_v1/index.html`.
Fill `outputs/diagnostics_sem/combined_review_v1/manual_review.csv` (16 rows: four models × four validation gap boxes).
The separate appearance review is `outputs/diagnostics_sem/dataset_final/appearance_review.csv`; its contact sheets are linked from `outputs/diagnostics_sem/dataset_final/index.html`.

Exact next command on this Windows machine:

```powershell
Start-Process 'D:\aksha\MTP\annotate_image.v4-train-test.coco\outputs\diagnostics_sem\combined_review_v1\index.html'
```

Completed reports were reused; no inference was repeated during the final continuation. The combined page was rendered in a headless browser and visually inspected. A read-only checker verified all five linked HTML pages, 101 local links/resources, 16 unique review rows, and explicit unknown labels for unavailable SSD candidate measurements:

```powershell
& .\outputs\mobilevitv2_ssd\env\Scripts\python.exe -B .\diagnostics\sem_defects\verify_report.py --combined outputs/diagnostics_sem/combined_review_v1
```

Use `review-summary` on the actual completed template:

```powershell
& .\outputs\mobilevitv2_ssd\env\Scripts\python.exe -B .\diagnostics\sem_defects\diagnose.py review-summary --review outputs/diagnostics_sem/combined_review_v1/manual_review.csv
```

To combine updated model reports later while retaining these manual edits, provide `--previous-review outputs/diagnostics_sem/combined_review_v1/manual_review.csv` and a **new** output directory. The reviewed fields are preserved by model/image/annotation ID; fields from individual model review files also survive when no previous combined row exists. Do not edit automated measurement columns.

This workflow inspects the existing reviewed **training and validation** data and trained model outputs. It never reads the test annotation file, runs test inference, trains, calibrates thresholds, changes labels, or updates baseline results. It imports existing preprocessing/decoding functions through adapters. No packages or checkpoints are downloaded by these commands.

Source files live here; generated artifacts belong under the already ignored `outputs/diagnostics_sem/`. Every command creates a new output directory and refuses an existing one. Run from the repository root. Use `-B` to avoid writing bytecode caches into imported baseline packages.

## What is available

| Experiment | Verified configuration | Local evidence |
|---|---|---|
| YOLOS unweighted | best checkpoint, 512 input, saved threshold 0.55 | trained safetensors, processor/config, selection/identity metadata, full validation saved query predictions |
| YOLOS weighted | best checkpoint, 512 input, saved threshold 0.60 | trained safetensors and same metadata types |
| SSD unweighted | Stage B epoch 38, 320 input, fp32, fixed diagnostic threshold 0.30 | run/selection metadata and postprocessed validation predictions; trained best.pt is absent locally |
| SSD weighted | Stage B epoch 38, 320 input, fp32, fixed diagnostic threshold 0.30 | run/selection metadata and postprocessed validation predictions; trained best.pt is absent locally |

Paths are explicitly configured in `configs/*.json` and verified against saved checkpoint/run metadata. Actual labels are resolved by name from COCO and model configuration; the current canonical names are bridge, gap, line_collapse and multi_bridge. YOLOS's no-object score is last; SSD's background score is first. Empty images remain zero-target images. SSD's 0.30 threshold is the already saved localization diagnostic threshold, **not** a newly selected or calibrated deployment threshold.

Current checked data: `data/prepared/label_studio/train.json` has 63 images, 44 boxes, 26 empty images, 3 gap source images / 6 gap instances. `val.json` has 16 images, 13 boxes, 5 empty images, 2 gap source images / 4 gap instances. Image paths resolve to `data/raw/images/`. Counts are calculated on each run, not hardcoded in the implementation. The split manifest is consulted for train/val groups and recorded fingerprints; the test split JSON is never opened or hashed by dataset inspection. Train and val file hashes and image hashes are recorded separately.

## Windows PowerShell commands

Use the existing environments without installing/upgrading them:

```powershell
$yolosPython = '.\venv\Scripts\python.exe'
$ssdPython = '.\outputs\mobilevitv2_ssd\env\Scripts\python.exe'
$diag = '.\diagnostics\sem_defects\diagnose.py'
$cfg = '.\diagnostics\sem_defects\configs'
```

1. Dataset-only inspection, original train/val contact sheets and appearance-review template:

```powershell
& $yolosPython -B $diag inspect --output outputs/diagnostics_sem/dataset_v1
```

2. YOLOS-unweighted validation smoke run (image 67 is a currently verified validation gap image):

```powershell
& $yolosPython -B $diag run --config "$cfg/yolos_unweighted.json" --image-ids 67 --device cpu --output outputs/diagnostics_sem/yolos_unweighted_smoke_v1
```

3. Full YOLOS-unweighted validation diagnosis:

```powershell
& $yolosPython -B $diag run --config "$cfg/yolos_unweighted.json" --device cpu --output outputs/diagnostics_sem/yolos_unweighted_v1
```

4. Other three runs (SSD saved-only mode is feasible without trained checkpoints):

```powershell
& $yolosPython -B $diag run --config "$cfg/yolos_weighted.json" --device cpu --output outputs/diagnostics_sem/yolos_weighted_v1
& $ssdPython -B $diag run --config "$cfg/ssd_unweighted.json" --mode saved --output outputs/diagnostics_sem/ssd_unweighted_v1
& $ssdPython -B $diag run --config "$cfg/ssd_weighted.json" --mode saved --output outputs/diagnostics_sem/ssd_weighted_v1
```

Saved-only mode does not manufacture anchor/query identities, full class distributions, or pre-filter candidates from final predictions. Per-gap A/B/C candidate searches are explicitly unavailable. It still generates exact preprocessing previews, geometry checks, operating-threshold final matching, and manual-review rows. The per-class score emitted by SSD is not necessarily its anchor's foreground argmax; that argmax cannot be recovered from these saved predictions.

5. Combine the four runs into one offline side-by-side index:

```powershell
& $yolosPython -B $diag combine --reports outputs/diagnostics_sem/yolos_unweighted_v1 outputs/diagnostics_sem/yolos_weighted_v1 outputs/diagnostics_sem/ssd_unweighted_v1 outputs/diagnostics_sem/ssd_weighted_v1 --output outputs/diagnostics_sem/combined_v1
```

6. Open the HTML:

```powershell
Start-Process .\outputs\diagnostics_sem\combined_v1\index.html
Start-Process .\outputs\diagnostics_sem\dataset_v1\index.html
```

The currently generated report paths are listed in RESUME.md. No server or internet access is needed. Keep the report directories together when transferring them: the combined page uses relative links to individual model reports. Do not transfer only `index.html`.

## Linux GPU commands

Copy this new diagnostics directory to the GPU repository before running it (no commit/push was performed during implementation). Use the **existing YOLOS environment** and the **existing separate SSD environment**, sequentially. Adjust `yolos_python` to `venv/bin/python` if that is where your YOLOS environment lives; do not point YOLOS at the SSD environment unless its dependencies are already present.

```bash
cd /home/ssl30/Music/mtp/MTP_ViT
yolos_python=.venv/bin/python
ssd_python=outputs/mobilevitv2_ssd/env/bin/python
diag=diagnostics/sem_defects/diagnose.py
cfg=diagnostics/sem_defects/configs

"$yolos_python" -B "$diag" inspect --output outputs/diagnostics_sem/dataset_gpu_v1
"$yolos_python" -B "$diag" run --config "$cfg/yolos_unweighted.json" --image-ids 67 --device cuda --output outputs/diagnostics_sem/yolos_unweighted_gpu_smoke_v1
"$yolos_python" -B "$diag" run --config "$cfg/yolos_unweighted.json" --device cuda --output outputs/diagnostics_sem/yolos_unweighted_gpu_v1
"$yolos_python" -B "$diag" run --config "$cfg/yolos_weighted.json" --device cuda --output outputs/diagnostics_sem/yolos_weighted_gpu_v1

# These must be the trained SEM checkpoints, not the downloaded COCO checkpoint:
ls outputs/mobilevitv2_ssd/natural_seed42_retry_20261027_230508/best.pt
ls outputs/mobilevitv2_ssd/weighted_seed42/best.pt
"$ssd_python" -B "$diag" run --config "$cfg/ssd_unweighted.json" --image-ids 67 --device cuda --output outputs/diagnostics_sem/ssd_unweighted_gpu_smoke_v1
"$ssd_python" -B "$diag" run --config "$cfg/ssd_unweighted.json" --device cuda --output outputs/diagnostics_sem/ssd_unweighted_gpu_v1
"$ssd_python" -B "$diag" run --config "$cfg/ssd_weighted.json" --device cuda --output outputs/diagnostics_sem/ssd_weighted_gpu_v1

"$yolos_python" -B "$diag" combine --reports outputs/diagnostics_sem/yolos_unweighted_gpu_v1 outputs/diagnostics_sem/yolos_weighted_gpu_v1 outputs/diagnostics_sem/ssd_unweighted_gpu_v1 outputs/diagnostics_sem/ssd_weighted_gpu_v1 --output outputs/diagnostics_sem/combined_gpu_v1
```

Each inference run uses `.eval()` and `torch.inference_mode()` with fp32, matching these saved baselines. Other baseline precision settings are explicitly rejected rather than silently changed. Nothing runs in parallel on the GPU. Source imports and YOLOS loading are local-only. SSD requires the already configured pinned CVNets checkout from the prior SSD setup; this workflow does not set up or patch it.

### Equivalent PowerShell commands for a Windows GPU host

These require an existing CUDA-enabled SSD environment, the configured CVNets source, and the **trained SEM** `best.pt` files. The current local CPU environment and missing trained checkpoints cannot execute these GPU steps. Do not substitute the downloaded COCO weights. On the original Linux GPU machine, use the Linux commands above instead.

```powershell
$ssdPython = '.\outputs\mobilevitv2_ssd\env\Scripts\python.exe'
$diag = '.\diagnostics\sem_defects\diagnose.py'
$ssdNatural = '.\diagnostics\sem_defects\configs\ssd_unweighted.json'
$ssdWeighted = '.\diagnostics\sem_defects\configs\ssd_weighted.json'

if (-not (Test-Path '.\outputs\mobilevitv2_ssd\natural_seed42_retry_20261027_230508\best.pt')) { throw 'Trained unweighted SSD checkpoint is missing' }
if (-not (Test-Path '.\outputs\mobilevitv2_ssd\weighted_seed42\best.pt')) { throw 'Trained weighted SSD checkpoint is missing' }

& $ssdPython -B $diag run --config $ssdNatural --image-ids 67 --device cuda --output outputs/diagnostics_sem/ssd_unweighted_gpu_smoke_v1
if ($LASTEXITCODE -ne 0) { throw 'SSD smoke check failed; inspect failure.json before proceeding' }
& $ssdPython -B $diag run --config $ssdNatural --device cuda --output outputs/diagnostics_sem/ssd_unweighted_gpu_v1
if ($LASTEXITCODE -ne 0) { throw 'Unweighted capture failed' }
& $ssdPython -B $diag run --config $ssdWeighted --device cuda --output outputs/diagnostics_sem/ssd_weighted_gpu_v1
```

After GPU captures have been copied back alongside the existing local YOLOS reports, build a **new** combined report preserving your review:

```powershell
& $ssdPython -B $diag combine --reports outputs/diagnostics_sem/yolos_unweighted_review outputs/diagnostics_sem/yolos_weighted_review outputs/diagnostics_sem/ssd_unweighted_gpu_v1 outputs/diagnostics_sem/ssd_weighted_gpu_v1 --previous-review outputs/diagnostics_sem/combined_review_v1/manual_review.csv --output outputs/diagnostics_sem/combined_review_with_ssd_raw_v1
```

The SSD adapter saves every decoded anchor's full five-class scores and identity once per image, plus per-class membership through floor filtering, top-k, NMS, maximum-detection limits, clipping validity and the existing operating threshold. It checks its stage replay against the official decoder on each captured image. Synthetic replay tests passed locally; real trained-SSD GPU capture and agreement with saved predictions remain unverified.

CLI overrides: `--family yolos|ssd`, `--experiment-id`, `--checkpoint`, `--run`, `--predictions`, `--train`, `--val`, `--image-root`, `--manifest`, `--device cpu|cuda`, `--image-ids ID ...`, and `--output`. Baseline thresholds and NMS settings are deliberately read from saved metadata; there is no threshold-search option. `--image-root` can be the repository root for stored relative paths, or the directory containing the image basenames. Ambiguous or missing paths fail.

## Candidate stages and matching

YOLOS captures every query's normalized model box, decoded original-coordinate box, foreground softmax scores and no-object score, preserving query ID. Baseline prediction export has no NMS, no confidence cutoff and no top-k beyond the model's 100 object queries. Operating-threshold filtering uses the foreground argmax score, even when no-object dominates, exactly like the baseline metrics. The processor uses the existing bilinear letterbox transform, 114 padding and its saved mean/std. The preview is reconstructed from the exact tensor supplied to the model, reversing that saved normalization, without image-specific contrast normalization.

SSD captures every decoded anchor with its normalized model box, baseline-clipped original-coordinate box and five softmax scores. Its lineage is tracked per (anchor, class): strict score floor >0.001, per-class top400, class-aware batched NMS IoU0.5, global max200, then the baseline's inverse mapping/clipping and zero-area filtering, followed by the existing operating threshold. The adapter verifies its replay against the unchanged official decoder and fails if they disagree. It records stage removal rather than inventing which specific neighbor suppressed an anchor. Raw normalized coordinates are retained before clipping. There is no upstream candidate truncation before the captured SSD raw forward in this implementation.

Candidate searches A/B/C50/C25 are **independent per ground truth** and may reuse one candidate. A finds best overlap regardless of class; B requires gap as foreground argmax (background dominance is separate); C finds highest gap score at IoU >=0.50 or >=0.25. IoU0.25 is only a loose diagnostic. A high-overlap candidate with tiny foreground probability is not automatically a useful detection. Full arrays are stored once per image in `candidates_image*.json.gz`; selected candidate records are small references/details in `per_gap.json`.

Final metrics use descending score, IoU >=0.50 and one-to-one matching. Gap precision includes predictions on **all selected validation images**, including clean/other-defect images. Full validation reports cover all16 images; smoke reports explicitly state that their denominators are partial. Class-agnostic gap localization matches all predicted classes to gap ground truths; its unmatched-prediction count is not the same as all-defect false positives. A separate all-class localization result is provided. No COCO AP is recomputed, so source annotation ID0 is untouched and cannot enter a COCO evaluator here.

Regenerated predictions are compared against saved validation predictions at 0.001 pixel box tolerance and 1e-5 score tolerance. Disagreement is prominent in HTML/Markdown and recorded numerically. YOLOS also reports an explicitly assumed positional query-order comparison because the saved exporter omitted query IDs. Saved-baseline final matching is reported separately. Do not relax tolerances or change thresholds to manufacture agreement. CPU/GPU/package differences may contribute, but a numerical discrepancy alone does not prove its cause.

## Manual review

Open the combined `index.html`, then the dataset contact sheets. Download/open `manual_review.csv` in a spreadsheet editor; it has one row per model and validation gap. `appearance_review.csv` covers original training and validation gap instances. These are plain CSV workflow templates generated by the diagnostic program, without spreadsheet formulas.

Fill only the manual columns:

- `annotation_correct`: yes/no/uncertain.
- `original_visibility`: clear/subtle/unclear.
- `input_visibility`: clear/subtle/lost/uncertain.
- `primary_failure`, `secondary_failure`: annotation_ambiguity, resolution_or_visibility_limitation, localization_failure, localization_imprecision, class_confusion, low_confidence, postprocessing_removal, instance_merging, successful_detection, uncertain.
- `reviewer_notes`, `reviewed` (yes when done).
- Appearance fields: single_or_adjacent_gaps, gap_size_relative_to_strip_width, pitch_orientation_differences, contrast_noise_differences, annotation_consistency, missing_training_appearances.

Inspect the **unannotated** crop as well as overlays. Labels/captions are outside the small defect regions; enlargement and interpolation are disclosed. Box width/height is not a measurement of the visible interruption width. Missing full candidate scores in SSD saved-only reports should remain uncertain; do not infer a suppression/classification cause from absent evidence. Review training appearances separately without moving validation images into training.

To summarize your edits without changing the review file:

```powershell
& $yolosPython -B $diag review-summary --review outputs/diagnostics_sem/combined_v1/manual_review.csv
```

To make a new report version while preserving manual fields:

```powershell
& $yolosPython -B $diag combine --reports outputs/diagnostics_sem/yolos_unweighted_v1 outputs/diagnostics_sem/yolos_weighted_v1 outputs/diagnostics_sem/ssd_unweighted_v1 outputs/diagnostics_sem/ssd_weighted_v1 --previous-review outputs/diagnostics_sem/combined_v1/manual_review.csv --output outputs/diagnostics_sem/combined_v2
```

Manual state joins on model + image ID + annotation ID. Existing reports/reviews are never overwritten. `review-summary` reports invalid/missing required fields even if a row is marked reviewed.

To regenerate a model's visuals from existing full captures without another forward pass:

```powershell
& $yolosPython -B $diag run --config "$cfg/yolos_unweighted.json" --mode captured --capture-report outputs/diagnostics_sem/yolos_unweighted_v1 --previous-review outputs/diagnostics_sem/yolos_unweighted_v1/manual_review.csv --output outputs/diagnostics_sem/yolos_unweighted_v2
```

This verifies captured checkpoint and annotation identities and reuses captured candidates. It does not recover missing raw data from a saved-only report. To share evidence, send the completed review CSV, `summary.md`, `summary.json`, `manifest.json`, `per_gap.json`, candidate dumps, and relevant original/input/crop PNGs. Include discrepancy reports and which models were saved-only. That evidence can inform a later experiment; this workflow itself does not recommend retraining or claim a root cause.

## Tests

No installation required. Standard-library unittest is used:

```powershell
& $ssdPython -B -m unittest discover -s diagnostics/sem_defects/tests -v
& $yolosPython -B diagnostics/sem_defects/tests/test_diagnostics.py -v -k yolos_tensor
```

The first command tests geometry, mappings, matching, adjacent instances, empty/missing cases, prediction comparison, SSD lineage against official postprocessing, and review preservation. It skips the YOLOS tensor check when Transformers is unavailable. The second checks exact tensor equality with the unchanged YOLOS dataset implementation in the YOLOS environment. See RESUME.md for actual execution status and limitations.
