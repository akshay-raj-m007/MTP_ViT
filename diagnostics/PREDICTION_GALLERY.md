# All-image prediction gallery

## Completed local inference — latest gallery

Both trained SSD checkpoints are now available, and **both models have completed fresh validation and test inference locally on CPU, fp32**. Open `outputs/prediction_gallery/all_models_complete_v1/index.html`. It contains all128 model/image views across16 validation and16 test images, using fresh SSD predictions and the existing YOLOS predictions.

```powershell
Start-Process 'D:\aksha\MTP\annotate_image.v4-train-test.coco\outputs\prediction_gallery\all_models_complete_v1\index.html'
```

Fresh SSD outputs and checkpoint provenance: `outputs/prediction_gallery/ssd_val_test_inference_v1/`. Existing training results and the earlier incomplete gallery were preserved. The earlier missing-checkpoint discussion below describes that previous state.

Commands used (choose fresh output names if rerunning):

```powershell
& .\outputs\mobilevitv2_ssd\env\Scripts\python.exe -B .\diagnostics\prediction_gallery.py capture-ssd --split both --device cpu --output outputs/prediction_gallery/ssd_val_test_inference_v1
& .\outputs\mobilevitv2_ssd\env\Scripts\python.exe -B .\diagnostics\prediction_gallery.py build --ssd-predictions-dir outputs/prediction_gallery/ssd_val_test_inference_v1 --output outputs/prediction_gallery/all_models_complete_v1
```

`capture-ssd --split val|test|both` supports either split or both. The older `capture-ssd-test` remains test-only. `--ssd-predictions-dir` and `--ssd-test-dir` are aliases. This performs inference only: no retraining, threshold tuning, or new test performance metrics.

## Original gallery and GPU instructions

Open `outputs/prediction_gallery/all_models_v1/index.html` on the current Windows machine:

```powershell
Start-Process 'D:\aksha\MTP\annotate_image.v4-train-test.coco\outputs\prediction_gallery\all_models_v1\index.html'
```

This separate workflow implements the user's explicit request to visualize validation **and test** predictions. It does not change the earlier validation-only gap diagnostic workflow. No metrics, threshold selection, training, labels or checkpoints are changed.

The gallery contains all16 validation and all16 test images, with one card per model. Predictions use original-image coordinates, a consistent class-color legend and the existing thresholds: YOLOS unweighted0.55, YOLOS weighted0.60, SSD0.30 (fixed diagnostic threshold, not calibrated). Hover boxes for labels/confidence; expand each card for coordinates. Ground-truth boxes are not drawn. Empty accepted predictions and unavailable predictions have different messages.

Locally available: all four validation prediction files and both YOLOS test files. Both trained SSD checkpoints and SSD test predictions are absent locally. The two SSD test cards therefore say unavailable, not “no detections.” Generated HTML/images/provenance are self-contained in the output folder and Git-ignored.

## Complete the missing SSD test views on Linux GPU

Copy the new `diagnostics/prediction_gallery.py` to the same repository path on the GPU machine. The trained SEM best.pt files must exist in the two configured run folders. Existing dependencies and pinned CVNets checkout are reused; nothing is installed or downloaded. Models run sequentially in eval/inference mode, fp32, with their unchanged postprocessing. This captures predictions for visualization, without computing test performance or changing parameters.

```bash
cd /home/ssl30/Music/mtp/MTP_ViT
ssd_python=outputs/mobilevitv2_ssd/env/bin/python

"$ssd_python" -B diagnostics/prediction_gallery.py capture-ssd-test \
  --device cuda \
  --output outputs/prediction_gallery/ssd_test_export_v1

"$ssd_python" -B diagnostics/prediction_gallery.py build \
  --ssd-test-dir outputs/prediction_gallery/ssd_test_export_v1 \
  --output outputs/prediction_gallery/all_models_complete_v1
```

Open the second output's `index.html` on Linux, or copy its **entire folder** back to Windows and open it there. It includes copied original images, so it does not depend on the repository image folder. Alternatively copy only `ssd_test_export_v1` back and rebuild locally:

```powershell
& .\outputs\mobilevitv2_ssd\env\Scripts\python.exe -B .\diagnostics\prediction_gallery.py build --ssd-test-dir outputs/prediction_gallery/ssd_test_export_v1 --output outputs/prediction_gallery/all_models_complete_v1
Start-Process .\outputs\prediction_gallery\all_models_complete_v1\index.html
```

All output directories must be fresh. No existing gallery, manual review or result is overwritten. A Windows GPU host can run the same capture command using its existing CUDA environment's `Scripts\python.exe`. The local environment is CPU-only; the trained SSD weights have now been pulled and local inference completed, as recorded above.

No commit or push was performed. If transferring through Git later, stage the new source separately and explicitly force-add only the four small exported JSON files if desired; do not add environments or pretrained caches.
