# Implementation notes

## Decisions and corrections

- The new input is COCO, not native Label Studio JSON. No additional native export was found in the project-level inputs. Therefore no guessed task-ID correspondence or percentage conversion was introduced.
- Canonical source-name remapping fixes the previous noncontiguous category/no-object collision. Exactly four foreground classes; the head produces a fifth internal no-object logit.
- `fig34` is clean (`no_defect`) by explicit user confirmation. It retains its training assignment, has zero boxes and uses `user_confirmed_clean` metadata. All 36 empty images count as clean; clean split counts are 26/5/5. Six gap image assignments were later changed at the user's request; box labels and geometry were unchanged.
- Natural sampling is baseline. Weighted sampling now uses image presence, exponent 0.5 and cap 3. Background exposure is explicitly reported, including the unannotated example.
- torchvision v2 (installed 0.28.0) provides maintained bounding-box-aware flips. Known-coordinate and empty-box tests exercise its real APIs. No extra augmentation package needed.
- Letterbox coordinates use actual integer resized width/height, so separate effective x/y scales are recorded. Inversion handles padding explicitly. Only predicted coordinates may be clipped to image bounds. Ground-truth boxes fail validation rather than being silently clipped or discarded.
- The processor is used with `do_resize=False`, `do_pad=False`; rescale/normalize happens once. Source images become RGB. Labels and original image identity are retained separately from canvas targets.
- Dedicated training and evaluation loops replace Trainer. There is no private sampler override to depend on. Stage optimizers and warmup schedules are rebuilt. Whole-split COCO `map` controls best-model and early-stop decisions, greater is better. Output tuples are never assumed batch-shaped.
- Predictions use all model queries for AP, original coordinates and image IDs. COCO evaluator state is freshly constructed each call. Fixed-threshold matching is score-ordered, same-class, one-to-one at IoU 0.5; wrong-class and duplicate predictions count correctly.
- Validation selects the global threshold; test and inference load it. Metadata fingerprints reject model/data mismatches in evaluation. Image classification is reported separately from localization.
- Prepared-data identity now hashes the actual train, validation and test COCO contents along with the source fingerprint. This prevents a checkpoint from the previous gap split from passing the identity check on the new split.
- The old visualization incorrectly used the filtered detection index to select a token. The new attention entry point selects from original logits and derives patch layout from actual token dimensions. Legacy detailed plots are archived, not silently run against a new checkpoint.

## Environment and executed checks

Existing Python 3.14.2, torch 2.13.0, torchvision 0.28.0, transformers 5.16.1, Pillow 12.3.0, pycocotools 2.0.11. Requirements pin installed package versions. CPU-only; cached official YOLOS-Tiny safetensors and processor loaded successfully. Model module names verified: `class_labels_classifier`, `bbox_predictor`, `vit.encoder.layer`, `vit.layernorm`.

Stage A: 149,961 trainable parameters. Stage B: 1,040,073 of 6,471,945 total. Earlier blocks, detection tokens and positional embeddings remain frozen.

Focused tests cover noncontiguous mapping, Windows basename resolution/ambiguity, empty and multiple targets, known flips and resize/normalization coordinates, inverse round trip, duplicate/group split isolation, perfect COCO AP, wrong classes, duplicate predictions, clean false positives, zero predictions and multilabel reporting. Results are saved in `outputs/checks/unit_tests.txt`.

Real 512x512 CPU smoke: Stage A empty image ID 10, loss 1.6541104; Stage B multi-object image ID 49, loss 2.8351715. Both backward and optimizer steps succeeded. Five validation images (one per foreground class plus clean) passed inference and metric calculation. Smoke AP=0.0 is not a trained-model performance claim. Full saved evidence: `outputs/checks/smoke_512_verified/report.json`. No test predictions or full training.

Five augmented previews were generated and visually inspected; boxes align with the transformed images. An initial smoke attempt exposed a NumPy `int64` in AP support serialization; fixed to Python int, added JSON serialization coverage, and successfully reran. The initial failed directory is retained for audit.

## Remaining limitations

- No native Choices/review status: 35 empty images are assumed clean, not independently verified. No external source groups supplied.
- 31 dHash candidate pairs (13 cross-split) need review. dHash is only screening; repeated SEM patterns create false positives and can hide related sources. Exact pixel duplicate groups are empty. Current evaluation cannot establish source-independent generalization until source relationships are resolved.
- Actual image sizes span 59 distinct shapes, with tiny 81x81 originals and 1024x1024 images. 512 is an initial experiment, not an optimum. No tiling or synthetic generation was introduced.
- CUDA, AMP, multi-worker execution, a complete training schedule, and interrupted/resumed training were not run. Resume code saves epoch-boundary state, but operational interruption recovery is unverified. Distributed training is rejected.
- Prepared annotations support validated non-crowd, non-ignored COCO bounding boxes. Native JSON, crowd/ignore policy and conflicting annotation reconciliation require explicit extensions.
- Normal checkpoint saving within a new run overwrites its own `best`, stage-best and `last` directories; existing run directories otherwise require `--resume`. Do not run two writers against one output directory. Checkpoint directory writes are not transactional across all files.
- Historical full architecture figures remain in `archive/scripts/legacy_visualization/`; updated visualization focuses on correct detection and final-layer attention. No claim that attention proves causal feature importance.

Sampling check: 10,000 seeded weighted draws produced 36.80% empty-image exposure versus expected 37.33%. Natural exposure is 26/63 = 41.27%. The expected assumed-clean share is 35.90%; user-confirmed clean share 1.44%. Weights range 1.0 to 2.16025. Evidence: outputs/checks/sampling_check.json. This checks sampler behavior, not model accuracy.


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
