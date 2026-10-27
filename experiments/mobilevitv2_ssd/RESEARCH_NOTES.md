# Implementation and provenance

## Official detector

- Source: [Apple CVNets](https://github.com/apple/ml-cvnets), revision `9b4d544e46011407dc12cde3735422b5eb8dc265`.
- [Official model zoo](https://github.com/apple/ml-cvnets/blob/9b4d544e46011407dc12cde3735422b5eb8dc265/docs/source/en/general/README-model-zoo.md): SSD with MobileViTv2 width multiplier 0.75. The actual class is `SingleShotMaskDetector`; this implementation uses its bounding-box detection path, not segmentation.
- [Complete COCO detector checkpoint](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-0.75.pt), 14,564,469 bytes; SHA256 `5538a12e1350bf06f05f29917c48caa8c0a1dab3a458727db06764b0f77df055`.
- [Matching official YAML](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-0.75.yaml), copied as `configs/upstream_coco.yaml`; SHA256 `202034974608062959017a5cb3501ea8ce6604b282a228237ee397108095cdf7`.
- The upstream Apple license is copied in [UPSTREAM_LICENSE](UPSTREAM_LICENSE). It is Apple's custom source license, not a blanket MIT license; some upstream SSD utilities separately attribute MIT-licensed sources. Refer to the upstream license and model hosting terms when redistributing source or weights.

The complete 81-class SSD loads strictly before adaptation: 3,584,955 original parameters, zero missing/unexpected keys. Six prediction convolutions interleave four offsets and class logits per anchor. Only the four new foreground channels are initialized; the four regression channels and COCO background channel for each anchor are copied exactly. The final detector has 2,979,235 parameters, of which 2,947,355 retain pretrained values and 31,880 are newly initialized class parameters. Backbone and detector feature layers are retained. Tests check the preserved channels against the official checkpoint tensor values.

MobileViTv2 is constructed through its upstream `build_model`, preserving classification-specific Swish activation while the SSD retains its configured ReLU. Single-device BatchNorm replaces synchronized BatchNorm with compatible parameters/buffers. The only edit made to the newly downloaded source redirects `common.TMP_CACHE_LOC` from `/tmp/cvnets` into the experiment's ignored cache via an environment variable. No original project files are patched.

## Input, matching, loss and decoding

Official evaluation input: RGB, bicubic direct resize to 320×320, tensor in [0,1], no ImageNet mean/std normalization. No classification processor is involved. Optional letterbox preserves effective per-axis scale and padding for inverse mapping. Resize and letterbox round trips are tested. Grayscale images are converted to RGB.

Default augmentation is independent horizontal/vertical flips with probability 0.5 each, brightness/contrast factors sampled from [0.9,1.1]. Geometry and boxes transform together. No shear, crop, mosaic, Mixup, CutMix, strong stretching or blur. Validation is deterministic. Training uses one on-the-fly view per sampled original, not a stored 20-view expansion.

Official anchors use feature strides 16,32,64,128,256 and the final pooled feature, scale range 0.1–1.05, aspect ratios 2 and 3 except the last feature (2). The six feature projections are 512,256,256,128,128,64. Anchors are generated from actual feature-map shapes. Verified forward shapes: 3,232 anchors at 320; 8,188 at 512. The upstream matcher uses IoU 0.5, forced best matches and center/size variances 0.1/0.2. An exact-anchor positive match and decode round trip are tested.

The established loss is SmoothL1 box regression plus softmax cross-entropy with 3:1 hard-negative mining and normalization by max(1,positive anchors). No class-weighted loss, focal loss, contrastive loss or adaptive localization weighting is added. **Explicit empty-image adaptation:** upstream mines zero negatives when an image has zero positives. This wrapper adds the three highest background cross-entropies for each empty image, divided by the same batch positive count denominator. Empty regression targets are finite zeros, avoiding upstream dummy zero-size boxes and log(0). Tests demonstrate upstream zero loss, wrapper positive loss/nonzero gradients on empty images, and identical loss to upstream on positive-only inputs.

Decode uses the upstream matcher and softmax followed by per-class filtering at score >0.001, top 400 candidates per class before class-aware torchvision batched NMS at IoU 0.5, then at most 200 detections per image across classes. Coordinates are mapped back and prediction boxes clipped to image bounds. Zero-area predictions after clipping are omitted; ground truth is never clipped or silently repaired. COCO AP uses its standard maxDets [1,10,100] and IoUs 0.50:0.05:0.95, over the whole validation set. Ground-truth annotation IDs are renumbered 1..N in an evaluator-local copy only. No per-epoch confidence-threshold/F1 optimization.

## Dataset inspected

Active data: `data/prepared/label_studio/`, manifest `split_manifest.json`, images resolved from `data/raw/images/`. Inspection of current YOLOS configs, dataset/path resolver, split fingerprint, transforms, sampler and corrected COCO metrics established this location; the old training entry point is never imported. Raw annotations remain `data/raw/result.json`.

Prepared fingerprint: `54e0ef39bfabb1cf4e849733f7d9a056878c95825c1c2532f0c9039bb919f276`.

| Split | Images | Boxes | Empty images |
|---|---:|---:|---:|
| Train | 63 | 44 | 26 |
| Validation | 16 | 13 | 5 |
| Test, structural inspection only | 16 | 12 | 5 |
| Total | 95 | 69 | 36 |

| Class | Training images | Training boxes | Validation boxes | Test boxes |
|---|---:|---:|---:|---:|
| bridge | 14 | 17 | 4 | 4 |
| gap | 3 | 6 | 4 | 3 |
| line_collapse | 14 | 15 | 3 | 3 |
| multi_bridge | 6 | 6 | 2 | 2 |

All three reviewed training gap images remain assigned to train: `baea786d-fig14.jpg`, `d40dc8e8-fig2.jpg`, `79c680ce-fig29.jpg`. Validation/test images are never sampled for training. Structural checks inspect all split memberships, references, dimensions, hashes, finite positive in-bounds boxes, duplicate IDs, file overlap and available source groups. Structural inspection does not establish that annotations are semantically complete, or prove source independence when source metadata is absent. Empty images are consumed as background according to the supplied reviewed dataset, including the deliberately unannotated clean example.

## Sampling and training

Weights use unique image class presence, not box frequency. A foreground image gets `min(3, max((largest_class_image_count / class_image_count)**0.5))`; the maximum handles multiclass images. Clean image weight is 1. Weighted sampling uses replacement, with exactly 63 draws per epoch. Natural sampling is a shuffled traversal of all 63 originals. Separate torch generators control sampling, augmentation seeds and loader worker seeds.

| Expected presentations per epoch | Natural | Weighted |
|---|---:|---:|
| bridge | 14.00 | 12.66 |
| gap | 3.00 | 5.86 |
| line_collapse | 14.00 | 12.66 |
| multi_bridge | 6.00 | 8.29 |
| Clean fraction | 41.27% | 37.33% |

Actual weighted counts fluctuate. Weight range is 1–2.16025. Weighting is not assumed to be superior.

Stage A trains all non-encoder detector parameters (SSD heads and detector-specific extra feature layers), 498,226 parameters; the entire encoder and its BatchNorm running statistics remain frozen. Stage B reloads the validation-best Stage A checkpoint, enables `encoder.layer_5` plus the same detector parameters (1,826,869 parameters), and rebuilds optimizer groups. Earlier encoder stages remain in eval mode. Full lists are saved per run.

Defaults: AdamW weight decay 1e-4; A 15 epochs at head LR 1e-4; B 40 epochs at head LR 5e-5 / final encoder stage LR 5e-6; warmup 5% of each stage's update opportunities followed by cosine decay; gradient clipping 1; microbatch 1, accumulation 8, workers 0. The last seven-microbatch group uses divisor 7, not 8. Accumulation averages normalized microbatch losses; it is not numerically identical to SSD's positive-anchor normalization on one large physical batch. BatchNorm behavior also differs from a large batch. Fixed budget and validation after every epoch are the default.

## Verification and limitations

Verified locally on Windows, separate Python 3.12 environment, torch 2.5.1+cpu and torchvision 0.20.1+cpu. Core dependency versions are pinned in the new requirements file; CUDA wheel installation is separate. Dependencies from upstream's unrelated audio/text training paths are not required because only the necessary official detector classes are imported.

Executed: strict official checkpoint load; actual dataset inspection; 320 and 512 forwards with matching anchors; 14 previews from seven sources including all training gap images; CPU smoke optimizer steps on gap, empty and multi-object targets, plus one validation-image inference/decode; frozen parameter and BatchNorm checks; unit tests for category mapping, transforms, sampling/RNG restoration, annotation-ID-zero perfect AP, wrong-class/duplicate/clean false positives, empty loss and preserved pretrained channels.

No full training, optional overfit training, final test evaluation, commit or push was performed. GPU execution, GPU memory usage, Linux runtime and end-to-end interrupted-training resume have not been exercised here. The implementation supports them, but the GPU smoke command should be run before training. No claim of SEM accuracy can be made from random new class channels or smoke steps.

Final continuation checks: 11 focused tests passed, including a gradient-accumulation comparison against physical group means and class-agnostic localization matching. The preview output now also includes seven original-size unmarked/annotated pairs. Validation adds class-agnostic localization recall at fixed confidence 0.3 and IoU 0.5, using score-ordered one-to-one matching on post-NMS predictions; only class-aware COCO mAP selects checkpoints. See RESUME.md for the latest execution status and next command.
