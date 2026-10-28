# Diagnostic workflow state

Created 2026-10-27. All implementation is new under diagnostics/sem_defects. Do not modify existing training/model/data files. This workflow is diagnosis only.

Implemented: config/provenance checks; train/validation dataset inspection; exact baseline transforms; tensor previews and original/context crops; all YOLOS query scores and SSD anchor-stage adapters; strict prediction comparison; per-gap searches; one-to-one matching; standalone HTML/PNG reports; four-model comparison; editable CSV templates; review preservation and summaries; captured-output rerendering; tests and GPU commands.

Completed as of 2026-10-28: dataset-only inspection; YOLOS unweighted smoke; full16-image YOLOS unweighted and weighted CPU inference; both SSD saved-prediction reports. These completed inference outputs were reused during continuation, not regenerated.

Final combined report: `outputs/diagnostics_sem/combined_review_v1/index.html`.
Manual template: `outputs/diagnostics_sem/combined_review_v1/manual_review.csv` (16 rows, all awaiting human review).
Training/validation appearance review: `outputs/diagnostics_sem/dataset_final/appearance_review.csv` and adjacent `index.html`.

Combined inputs: `yolos_unweighted_review`, `yolos_weighted_review`, `ssd_unweighted_review`, `ssd_weighted_saved`, all below `outputs/diagnostics_sem/`. Do not overwrite them. The earlier `combined_review/` directory contains a failed initial combine attempt (missing import); it is preserved. The corrected usable report is **combined_review_v1**.

Fixed within diagnostics only: missing `read` import in combined reporting; preservation of per-model manual fields when no previous combined row exists; explicit `unknown_not_captured` status for SSD raw evidence; added combined-review round-trip regression test and read-only report verifier.

Executed final checks: standard unittest suite (14 tests: 13 passed, YOLOS tensor check skipped in SSD environment because Transformers is absent); the separate exact YOLOS-tensor comparison passed in the YOLOS environment, so all14 focused checks passed across the two environments. Synthetic SSD lineage agrees with official CVNets postprocessing. Edited reviewer notes including comma/Unicode and review statuses survive combined-report regeneration while original source/review files stay byte-identical. The read-only report checker passed: 5 HTML pages, 101 local links/resources, 16 unique rows. Combined HTML was rendered in headless Chrome and visually inspected. Review-summary reports 0 reviewed /16 pending. Generated report and CSV paths are Git-ignored.

Final protection audit: all421 pre-existing tracked files matched `outputs/diagnostics_sem/original_tracked_hashes.json`. Tracked git diff is empty; only the new `diagnostics/` source directory is untracked. Existing model code, images, annotations, splits, checkpoints, predictions and results were unchanged. No completed inference was repeated in the final continuation.

Known limitations: trained SSD best.pt files are absent locally, so SSD raw inference is unexecuted; saved-only reports cannot reveal full class scores or suppression history. YOLOS CPU regeneration differs from saved GPU predictions under the strict comparison tolerances; reports flag this, retain provenance and separate saved-baseline matching. Do not claim a definitive root cause. No test-set inference/analysis, training, package changes, model downloads, commits or pushes have occurred.

Exact next command locally:

```powershell
Start-Process 'D:\aksha\MTP\annotate_image.v4-train-test.coco\outputs\diagnostics_sem\combined_review_v1\index.html'
```

Next command for GPU SSD smoke, after copying this diagnostics directory to the repository there:

```bash
outputs/mobilevitv2_ssd/env/bin/python -B diagnostics/sem_defects/diagnose.py run --config diagnostics/sem_defects/configs/ssd_unweighted.json --image-ids 67 --device cuda --output outputs/diagnostics_sem/ssd_unweighted_gpu_smoke_v1
```

For full commands, review fields and preservation procedures, read README.md.
