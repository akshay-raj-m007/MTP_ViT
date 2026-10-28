Project summary: SEM defect detection using YOLOS and SSD–MobileViTv2
1. Problem statement
The project investigates automatic detection and classification of semiconductor manufacturing defects in scanning electron microscope (SEM) images.

The images contain repeated line or grating structures. The objective is to predict an axis-aligned bounding box around every visible defect and assign one of four defect classes:
Model ID	Class	Meaning
0	bridge	An unwanted connection between adjacent lines
1	gap	A local interruption or missing segment in a line
2	line_collapse	A deformed, shifted, merged, or collapsed line structure
3	multi_bridge	Multiple adjacent bridge connections within one defect region


Images with no annotated defect are treated as no_defect or clean images. no_defect is not a foreground bounding-box class. Clean images contain zero boxes and are used as background examples.

The main research question is whether lightweight pretrained object detectors can learn these four visually subtle SEM defects from a very small, imbalanced dataset, and whether class-weighted sampling improves rare-defect performance.

The project compares two detector architectures and two sampling strategies, producing four trained models:
1. YOLOS-Tiny with natural sampling.
2. YOLOS-Tiny with weighted sampling.
3. SSD–MobileViTv2 with natural sampling.
4. SSD–MobileViTv2 with weighted sampling.
Repository:

https://github.com/akshay-raj-m007/MTP_ViT.git
2. Image collection and provenance
The working dataset contains 95 original SEM images under:
data/raw/images/
The original annotation export is:
data/raw/result.json
The filenames have forms such as:
43e08f23-fig5.jpg
baea786d-fig14.jpg
This naming suggests that the files were extracted or collected as numbered figures, but the repository does not contain authoritative provenance mapping each image to its publication, acquisition session, wafer, process condition, or microscope capture.

Therefore, the following facts remain unknown from the repository alone:
- The original paper or acquisition source of every image.
- Whether several images came from the same SEM session or specimen.
- Whether visually similar images are crops from the same original figure.
- Whether acquisition conditions are independent across train, validation, and test.
Older Roboflow exports are preserved under archive/, but the active pipeline does not use those exports. The active source is the 95-image collection and the COCO annotation file above.

This lack of source-group metadata is an important limitation. The similarity checker found 31 near-duplicate candidates, including 13 candidates crossing the current split boundaries. These are only similarity hints, not confirmed duplicates. Without manually identifying their original sources, source-independent generalization cannot be established.
3. Annotation process and data preparation
The active annotation input is a COCO-format export associated with the Label Studio workflow. It is not a native Label Studio task export containing review-state metadata.

Each foreground annotation contains:
- Image ID.
- Annotation ID.
- Category ID.
- Axis-aligned COCO bounding box.
- Area and related COCO fields.
The prepared files are:
data/prepared/label_studio/train.json
data/prepared/label_studio/val.json
data/prepared/label_studio/test.json
data/prepared/label_studio/split_manifest.json
data/prepared/label_studio/class_mapping.json
data/prepared/label_studio/validation_report.json
The data preparation code is:
scripts/prepare_data.py
It performs the following checks and conversions:
- Resolves image paths by exact filename.
- Maps classes by their names rather than assuming unsafe category ordering.
- Confirms image files and dimensions.
- Validates bounding-box geometry.
- Rejects invalid or unknown classes.
- Checks unique image and annotation IDs.
- Detects duplicate or potentially related images.
- Records file hashes and a dataset fingerprint.
- Preserves existing split assignments unless an explicit resplit is requested.
Annotation ID zero correction
An earlier validation annotation used COCO annotation ID 0. COCOeval also uses zero as the unmatched sentinel, so that annotation could be matched incorrectly.

The current preparation and evaluation implementations correct this by assigning positive annotation IDs in prepared data and independently renumbering annotations in an evaluator-local copy. The raw result.json remains unchanged.

Current results use the corrected evaluation implementation and current gap split.
Clean images
There are 36 zero-box images:
- 26 in training.
- 5 in validation.
- 5 in test.
One image, c5329556-fig34.jpg, was explicitly confirmed as deliberately unannotated and clean.

The remaining zero-box images are assumed clean from the supplied COCO dataset. Because native review metadata was unavailable, their semantic cleanliness still depends on human visual inspection.
4. Dataset statistics
The complete dataset contains:
Item	Count
Images	95
Foreground boxes	69
Clean images	36
Defect classes	4


Full-dataset box counts
Class	Boxes	Images containing class
bridge	25	22
gap	13	7
line_collapse	21	20
multi_bridge	10	10
no_defect	0	36


The major imbalance is the gap class: only seven independent source images contain gaps.
Current split
Split	Images	Boxes	Clean images
Train	63	44	26
Validation	16	13	5
Test	16	12	5


Per-class box distribution
Class	Train	Validation	Test
bridge	17	4	4
gap	6	4	3
line_collapse	15	3	3
multi_bridge	6	2	2


Per-class image distribution
Class	Train	Validation	Test
bridge	14	4	4
gap	3	2	2
line_collapse	14	3	3
multi_bridge	6	2	2
no_defect	26	5	5


Current gap split
Training:
baea786d-fig14.jpg
d40dc8e8-fig2.jpg
79c680ce-fig29.jpg
Validation:
a3c2b033-fig26.jpg
d799ade3-fig45.jpg
Test:
b5e1d95a-fig35.jpg
add3881f-fig91.jpg
The gap split was manually changed because some earlier training images contained several adjacent gaps, while validation and test included single-gap appearances. The aim was to reduce this appearance mismatch, but the current results show that gap detection is still effectively unsuccessful.
5. Manual annotation-review workflow
The local review tool is:
scripts/review_annotations.py
Run on Windows:
& .\venv\Scripts\python.exe .\scripts\review_annotations.py
Run on Linux:
.venv/bin/python scripts/review_annotations.py
It creates an ignored output directory containing:
- index.html
- review.csv
- Full-resolution images with annotation overlays.
- Marked and unmarked defect crops.
- Clean-image views.
- Deterministic preprocessing previews.
- Training augmentation previews.
- A structural validation report.
The reviewer should manually check:
1. Every foreground box contains the intended defect.
2. The class label matches the visible defect.
3. No visible defect is missing a box.
4. Boxes are neither too tight nor unnecessarily large.
5. Multiple nearby defects are annotated consistently.
6. Every zero-box image is truly clean.
7. Gap examples are visually consistent across splits.
8. Similar images across splits are not crops or variants of the same source.
The automated checks can validate IDs, paths, geometry and counts, but cannot prove annotation completeness or semantic correctness.

A second diagnostic workflow exists under:
diagnostics/sem_defects/
Its combined validation review page is:

[Combined diagnostic review](D:/aksha/MTP/annotate_image.v4-train-test.coco/outputs/diagnostics_sem/combined_review_v1/index.html)

The manual review CSV is:

[manual_review.csv](D:/aksha/MTP/annotate_image.v4-train-test.coco/outputs/diagnostics_sem/combined_review_v1/manual_review.csv)

It has one row per validation gap instance and model. Manual fields record annotation correctness, original/input visibility, likely failure category and reviewer notes.

At the last recorded diagnostic state, all 16 rows were still awaiting human completion.
6. Augmentation used for YOLOS
YOLOS training uses one original view plus 19 augmented views for each of the 63 training images.

The augmentations are generated on demand; augmented files are not permanently added to the dataset.

Each source image has:
Transformation	Views
Horizontal, vertical and combined flips	3
Width/height stretching and compression	4
Rotations at −12° and +12°	2
Horizontal and vertical shear	2
Padded scaling at 0.8× and 1.15×	2
Gaussian noise with σ = 0.01, 0.03 and 0.05	3
Combined geometric/noise/intensity transformations	3
Original deterministic view	1
Total	20


This gives:
63 source images × 20 views = 1,260 presentations per epoch
Images and bounding boxes undergo the same geometric transformations. Rotation and shear produce new axis-aligned enclosing boxes around the transformed defect. Box area is recalculated.

Validation and test images receive deterministic preprocessing only.

Important qualification: the 1,260 presentations are not 1,260 independent SEM images. They are transformed variants of only 63 source images.
7. Model 1: YOLOS-Tiny with natural sampling
Architecture:
hustvl/yolos-tiny
YOLOS is a transformer-based, end-to-end object detector. It uses fixed object queries and does not require conventional anchor generation or NMS in the project’s evaluation path.

Reasons for selecting it:
- A pretrained COCO checkpoint is available.
- The tiny version is feasible on limited hardware.
- It provides a transformer-based detector for comparison with SSD.
- Its query-based output provides a relatively direct detection pipeline.
- It can be adapted to four SEM foreground classes.
The adapted model has approximately 6.47 million parameters. Its final class head is reinitialized because COCO and SEM have different class counts. The COCO head-size mismatch message during loading is expected.

Training setup:
- Input: 512×512 letterboxed image.
- Seed: 42.
- Microbatch: 1.
- Gradient accumulation: 8.
- Stage A: detector heads, up to 15 epochs.
- Stage B: detector heads plus last two encoder blocks, up to 40 epochs.
- AdamW.
- Stage A learning rate: 1e-4.
- Stage B head learning rate: 5e-5.
- Stage B encoder learning rate: 5e-6.
- Weight decay: 1e-4.
- Gradient clipping: 1.
- Warmup: 5%.
- Natural sampling visits every source/recipe combination once per epoch.
- Validation mAP selects the best checkpoint.
- Best checkpoint: Stage B epoch 9.
- Selected confidence threshold: 0.55.
8. Model 2: YOLOS-Tiny with weighted sampling
This model uses the same architecture, input resolution, augmentation recipes, optimizer and training stages as Model 1.

The difference is weighted sampling with replacement.

Weights are based on class presence in each source image:
- gap receives the strongest promotion.
- multi_bridge receives moderate promotion.
- Bridge, line collapse and clean images receive lower relative weight.
The balanced YOLOS configuration uses:
sampling_power = 1.0
sampling_cap = 5.0
Approximate per-image weights are:
Class	Approximate weight
bridge	1.00
gap	4.67
line_collapse	1.00
multi_bridge	2.33
clean	1.00


The sampler still draws 1,260 presentations per epoch, but because it samples with replacement:
- Some source/augmentation views can repeat.
- Some views may not appear in an epoch.
- The original view of every image is no longer guaranteed.
Best checkpoint: Stage B epoch 13.

Selected confidence threshold: 0.60.
9. Models 3 and 4: SSD–MobileViTv2
The second architecture is an SSD detector with a MobileViTv2-0.75 backbone, based on Apple’s pinned CVNets implementation and official COCO detector checkpoint.

Architecture source:
Apple CVNets, revision 9b4d544e46011407dc12cde3735422b5eb8dc265
SSD–MobileViTv2 width multiplier 0.75
Reasons for selecting it:
- It is compact and suitable for deployment-oriented comparison.
- MobileViTv2 combines convolutional processing with lightweight global representation.
- SSD provides an anchor-based contrast to YOLOS’s query-based detector.
- An official pretrained COCO detector checkpoint is available.
- It has substantially fewer parameters than YOLOS.
The adapted detector contains approximately 2.98 million parameters. About 2.95 million retain pretrained values, while 31,880 new foreground-class parameters are initialized.

SSD internal class IDs are:
0 = background
1–4 = four SEM defect classes
Training setup:
- Input: direct resize to 320×320.
- Seed: 42.
- FP32 CUDA training.
- Microbatch: 1.
- Gradient accumulation: 8.
- Stage A: detector and SSD feature layers, 15 epochs.
- Stage B: same components plus encoder.layer_5, 40 epochs.
- AdamW and the same main learning-rate schedule as YOLOS.
- Horizontal/vertical flips and mild brightness/contrast augmentation.
- Three empty-image hard-background samples.
- No YOLOS-style 20-view recipe schedule.
- Class-aware NMS at IoU 0.5.
- Score floor: 0.001.
- Pre-NMS top-k: 400 per class.
- Maximum detections: 200.
- Best checkpoint selected by validation COCO mAP.
This means the YOLOS-versus-SSD comparison is not a perfectly controlled architecture-only experiment. Input resolution, augmentation exposure and postprocessing differ.
SSD natural model
- 63 presentations per epoch.
- Each training image visited once per natural epoch.
- Best checkpoint: Stage B epoch 38.
SSD weighted model
- 63 draws per epoch with replacement.
- Mild weighting uses sampling_power=0.5 and cap 3.
- Expected gap presentations rise from 3 to approximately 5.86 per epoch.
- Best checkpoint: Stage B epoch 38.
10. Current results
Validation comparison
Model	mAP@0.50:0.95	AP@0.50	Gap AP	Gap AP@0.50
YOLOS natural	25.65%	42.11%	0.40%	1.07%
YOLOS weighted	24.37%	42.96%	0.13%	0.21%
SSD natural	8.68%	19.48%	0.00%	0.00%
SSD weighted	9.77%	19.94%	0.00%	0.00%


Validation AP by class
Model	Bridge AP	Gap AP	Line-collapse AP	Multi-bridge AP
YOLOS natural	31.02%	0.40%	22.61%	48.58%
YOLOS weighted	24.52%	0.13%	17.29%	55.52%
SSD natural	10.51%	0.00%	0.84%	23.37%
SSD weighted	10.36%	0.00%	2.02%	26.68%


SSD class-agnostic localization recall at confidence 0.30 and IoU 0.50 was:
SSD model	Matched validation boxes	Recall
Natural	4/13	30.77%
Weighted	5/13	38.46%


This indicates that weighted SSD localized one additional box regardless of predicted class, but it still did not produce correct gap detections.
YOLOS final test results
Model	Test mAP	Test AP@0.50	Detection F1	Classification macro-F1
YOLOS natural	22.57%	38.56%	38.46%	53.71%
YOLOS weighted	25.72%	41.08%	34.78%	45.33%


YOLOS test AP by class
Model	Bridge AP	Gap AP	Line-collapse AP	Multi-bridge AP
Natural	24.28%	0.00%	13.42%	52.60%
Weighted	27.78%	0.00%	21.24%	53.86%


YOLOS operating-threshold detection results
Natural test model at threshold 0.55:
- Precision: 35.71%.
- Recall: 41.67%.
- F1: 38.46%.
- True positives: 5.
- False positives: 9.
- False negatives: 7.
- Gap recall: 0/3.
- Clean specificity: 100%.
Weighted test model at threshold 0.60:
- Precision: 36.36%.
- Recall: 33.33%.
- F1: 34.78%.
- True positives: 4.
- False positives: 7.
- False negatives: 8.
- Gap recall: 0/3.
- Clean specificity: 100%.
Interpretation
The strongest overall detector is currently YOLOS. Weighted YOLOS improved test mAP, bridge AP, line-collapse AP and multi-bridge AP slightly, but reduced operating-threshold recall, detection F1 and image-level macro-F1.

Weighted sampling therefore did not provide a uniform improvement.

The SSD weighted run improved validation mAP from 8.68% to 9.77% and class-agnostic recall from 30.77% to 38.46%, but remained much weaker than YOLOS.

All four models failed on the gap class. The YOLOS gap AP is numerically near zero on validation and exactly zero on test. Both SSD models have zero validation gap AP.

SSD test predictions have been generated and visualized, but a formal SSD test metrics report has not been produced. Test metrics should therefore not be invented or inferred from the visualization.
11. Detection versus classification
All four architectures are object detectors. They predict per-box classes.

YOLOS also has a derived image-level classification report. This is calculated from the detections for each image; it is not a separately trained image classifier.

Consequently:
- A missed box can make a defective image appear clean.
- A wrongly classified box changes the image-level label.
- no_defect means no accepted defect detection at the selected threshold.
- Image-level classification metrics do not demonstrate an independent classification capability.
The SSD experiment currently reports bounding-box detection and per-box classes. Whole-image classification and calibrated deployment-threshold selection were deferred.
12. Visualization and diagnostics
A complete visual comparison of all four models is available here:

[Four-model validation/test gallery](D:/aksha/MTP/annotate_image.v4-train-test.coco/outputs/prediction_gallery/all_models_complete_v1/index.html)

It contains:
- 16 validation images.
- 16 test images.
- Four model views per image.
- 128 model/image views in total.
- Predicted bounding boxes.
- Class colors.
- Confidence scores and coordinates.
- No missing model views.
The page draws prediction boxes only. It does not draw ground-truth boxes.

YOLOS uses its validation-selected thresholds:
- Natural: 0.55.
- Weighted: 0.60.
SSD uses the existing fixed diagnostic threshold of 0.30. This is not a calibrated deployment threshold, so confidence counts should not be compared directly across YOLOS and SSD.
13. Main bottlenecks and open research issues
13.1 Extremely small independent dataset
Only 63 independent images are available for training. Augmentation creates additional presentations but cannot provide new defect morphology, microscope conditions or wafer variation.
13.2 Severe gap-class scarcity
Only three training images contain gaps, although they contain six boxes. This is insufficient to cover variation in:
- Single versus adjacent gaps.
- Gap width and height.
- Contrast.
- Strip pitch.
- Orientation.
- Noise.
- Local line geometry.
- Magnification and resolution.
Weighted sampling repeats these same three sources. It does not create new gap appearances.
13.3 Domain appearance mismatch
The gap class may differ between training and validation/test in number of adjacent gaps, scale, contrast or local pattern. The split was modified to reduce this mismatch, but gap performance remained near zero.
13.4 Possible annotation ambiguity
Small line interruptions can be difficult to distinguish from:
- Noise.
- Blur.
- Normal line-width variation.
- Line collapse.
- Bridge boundaries.
- Compression or interpolation artifacts.
Manual semantic review is still essential.
13.5 Low native resolution
Some originals are as small as 81×81 pixels. Resizing them to 320 or 512 pixels cannot recover missing detail. Small defects may become visually weak or interpolation-dependent.
13.6 Near-duplicate and source leakage uncertainty
There are 31 near-duplicate candidates, including 13 crossing splits. Because source groups are unavailable, the independence of validation/test estimates is uncertain.
13.7 Very small validation and test supports
Examples include:
- Four validation gap boxes.
- Three test gap boxes.
- Two multi-bridge boxes in each held-out split.
One correct or incorrect detection can change the reported metrics substantially. A single split and seed cannot establish stable model ranking.
13.8 Augmentation realism
Strong geometric stretching, rotation and shear transform the boxes correctly, but some generated SEM appearances may be physically unrealistic.

Axis-aligned boxes around rotated defects also contain additional background.
13.9 Sampling is not sufficient to solve missing diversity
Weighted YOLOS heavily promotes gap images, yet gap AP remains approximately zero. This suggests that the bottleneck is not simply how often the current gap images are sampled.
13.10 Architecture comparison is confounded
YOLOS and SSD differ in:
- 512 versus 320 input resolution.
- 1,260 versus 63 presentations per epoch.
- Strong recipe-based versus mild stochastic augmentation.
- Query-based versus anchor/NMS postprocessing.
- Model capacity.
- Preprocessing.
- Fine-tuning behavior.
The current experiment compares complete pipelines rather than isolating architecture alone.
13.11 Overfitting and unstable validation
YOLOS training loss continued falling after validation mAP peaked. The best natural checkpoint occurred at Stage B epoch 9, while later epochs produced lower validation mAP. Weighted YOLOS showed similar instability.

More epochs alone are therefore unlikely to reliably solve the problem.
13.12 Threshold and confidence calibration
YOLOS thresholds were selected on validation by detection F1. SSD currently uses a fixed 0.30 diagnostic threshold. Confidence values are not calibrated across architectures.
13.13 Remaining diagnostic uncertainty
Saved SSD predictions are available, but the complete raw anchor-level lineage has not been captured from the trained model on the original GPU environment. Therefore, missed SSD gaps cannot yet be definitively attributed to:
- Poor localization.
- Background dominance.
- Wrong-class scores.
- Low confidence.
- NMS removal.
- Candidate truncation.
Unavailable evidence should remain classified as unknown.
14. Current research conclusion
The project demonstrates that YOLOS-Tiny can learn bridge, line-collapse and multi-bridge patterns to a limited extent from this small SEM dataset. Multi-bridge is currently the strongest class. SSD–MobileViTv2 performs substantially worse under the current setup, though its weighted run gives a small improvement over its natural run.

The central unresolved problem is gap detection. Every trained model fails to detect gaps reliably, including models that oversample the gap class. This points toward insufficient independent gap examples, incomplete appearance coverage, limited resolution or ambiguous annotation rather than a simple lack of training exposure.

The results are preliminary because the dataset is extremely small, class supports are low, source independence is unverified and only one principal split and seed have been evaluated. The current metrics are appropriate as experiment records, but they are not yet evidence of deployment-ready SEM defect detection.