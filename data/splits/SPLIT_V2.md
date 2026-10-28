# Split v2 (grouped)

`split_v2.json` assigns every image to a split by filename and lists the groups that must stay together. `scripts/prepare_data.py` (default `--assign data/splits/split_v2.json`) refuses to run if any group crosses splits or if any image is missing or unknown.

## Why groups
Many images show the **same physical defect** at a different zoom or crop. For example, fig16, fig21 and fig38 are one bridge, and fig83–fig89 are one line collapse, some views rotated or zoomed out with grey padding. If one copy is in train and another is in val or test, the model is evaluated on a defect it already saw.

The old split had several such leaks:
- fig14 (train) and fig45 (val).
- fig16 and fig38 (train) with fig21 (test).
- fig83 and fig84 (val) with fig85–fig89 (train).

So the old val and test scores were optimistic.

The groups were found by side-by-side visual review of every defect image, with dHash ≤ 5 screening for near-identical clean renders.

| Group | Images | Split |
|---|---|---|
| bridge_A | fig16, fig21, fig38 | train |
| bridge_B | fig17, fig22, fig39 | val |
| multi_C | fig18, fig23, fig40 | train |
| multi_D | fig19, fig24, fig41 | val |
| collapse_E | fig20, fig25, fig42 | test |
| bridge_F | fig27, fig30, fig32, fig36, fig37 | train |
| collapse_G | fig28, fig31, fig33 | train |
| gapvoid_H | fig26, fig35 | train |
| bridge_I | fig12, fig43 | test |
| collapse_J | fig13, fig44 | val |
| gapfaint_K | fig14, fig45 | train |
| bridge_L | fig92, fig95 | train |
| bridge_M | fig4, fig93, fig94 | train |
| collapse_N | fig81, fig82 | train |
| multi_O | fig10, fig11 | train |
| collapse_P | fig83–fig89 | train |
| clean groups | {53,54} test; {55,57} val; {64,67,70,73,76}, {65,66,71,74,78}, {68,69,75,79} train | |

Single images:
- **Val:** fig1 (bridge), fig29 (gap), fig90 (line_collapse); clean fig47, fig49, fig61.
- **Test:** fig5 (multi_bridge), fig7 (bridge), fig91 (gap); clean fig50, fig52, fig72.
- **Train:** everything else.

## Result
| Split | Images | Clean | bridge | gap | line_collapse | multi_bridge (boxes) |
|---|---|---|---|---|---|---|
| train | 66 | 26 | 18 | 9 | 14 | 6 |
| val | 16 | 5 | 4 | 4 | 3 | 3 |
| test | 13 | 5 | 3 | 1 | 4 | 1 |

## Gap choices
There are only 5 independent gap defects: the void pair (fig26/fig35), the faint-dash pair (fig14/fig45), fig2, fig29 and fig91.
- Train gets both paired types plus fig2, so it covers the void, faint-dash and zoomed broken-line appearances.
- Val gets fig29, which has 4 boxes, to give the most stable checkpoint and threshold selection.
- Test gets fig91, a broken line like fig2 but a different defect.

Gap val and test numbers rest on one image each and will be noisy. Grouped cross-validation over these 5 gap sources is the way to get a stable gap estimate.

## Limits
Val and test measure new defects under imaging conditions also seen in training (the same micrograph series), not a new tool or wafer.
