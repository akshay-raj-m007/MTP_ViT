# Local manual annotation review

From the repository root, run:

```powershell
& .\venv\Scripts\python.exe .\scripts\review_annotations.py
```

On Linux, use `.venv/bin/python scripts/review_annotations.py`.
The command reads the prepared `train.json`, `val.json`, and `test.json` and the source
images. It creates a unique ignored directory under `outputs/manual_annotation_review/`
and prints the location of its `index.html`, `review.csv`, and validation report.
It does not alter source data or use a model.

Open `index.html` locally. Examine every full-resolution overlay and unmarked image,
then each unmarked and marked crop. For small defects, use the crops rather than the
contact sheets. The comparison panel can magnify pixels; the manifest records the
native crop coordinates and display scale. Empty images require explicit visual
review before calling them clean. Existing annotation status is shown separately from
the new, initially `UNREVIEWED`, review decision fields in `review.csv`.

The bundle includes deterministic letterbox preprocessing previews for every readable
image and configured training augmentation views for every gap training image plus
one representative bridge, line collapse, multi bridge, and empty training image.
Use `--all-training-augmentations` to preview all training images. `--seed` fixes
stochastic appearance effects in those previews. The augmentations are generated
from the same recipes and box transforms as training; validation and test images
receive only deterministic preprocessing.

Useful options:

```powershell
& .\venv\Scripts\python.exe .\scripts\review_annotations.py --crop-padding 2 --min-crop-display 400 --seed 42
& .\venv\Scripts\python.exe .\scripts\review_annotations.py --no-augmentations
& .\venv\Scripts\python.exe .\scripts\review_annotations.py --all-training-augmentations
```

The structural report checks counts, image files and dimensions, class mapping,
unique IDs, orphan annotations, box geometry, annotation ID zero, and repeated
originals across splits. These checks cannot confirm that every visible defect was
boxed or that a class is semantically correct. If optional Torch dependencies are
unavailable, the base overlay, crop, HTML, CSV, and structural report are still
generated; the report states which previews were unavailable.
