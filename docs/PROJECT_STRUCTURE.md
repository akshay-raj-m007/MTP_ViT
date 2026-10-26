# Project structure and migration

All paths below are relative to the project root. Run documented PowerShell commands from that directory.

| Previous location | Current location |
|---|---|
| Root Python command files | `scripts/` |
| `sem_yolos/` | `sem_yolos/` (shared package) |
| Root `pipeline_config*.json` | `configs/` |
| `images/` | `data/raw/images/` |
| `result.json` | `data/raw/result.json` |
| `prepared_label_studio/` | `data/prepared/label_studio/` |
| `runs/` | `outputs/runs/` |
| `checks/` | `outputs/checks/` |
| `READme.md` | `docs/USAGE.md` |
| Dataset, implementation and augmentation documents | `docs/` |
| `C.md` | `docs/notes/C.md` |
| Presentation source | `docs/presentations/` |
| Old Roboflow `train/`, `valid/` and dataset readmes | `archive/roboflow/` |
| Old `yolos-sem-defect*` checkpoint folders | `archive/checkpoints/` |
| Old results, visualization output and predictions | `archive/results/` |
| `legacy_visualization/` | `archive/scripts/legacy_visualization/` |

`venv/` stays at its original location because Python environments can contain absolute paths. The package uses a flat layout: reusable modules live in `sem_yolos/`, while executable commands live in `scripts/`. Direct script execution adds the project root to the import path; package installation is not required.

## Paths and existing checkpoints

Configuration defaults now use `configs/` and `data/prepared/label_studio/`. Prepared COCO files and manifest image paths now point to `data/raw/images/`. Original exported annotations remain byte-for-byte unchanged; preparation resolves their exported Windows paths by exact image basename.

Existing saved checkpoints may still contain `prepared_label_studio` in their internal `pipeline_config.json`. The shared path resolver maps that old name to the current prepared-data folder. Resume comparison normalizes this relocation while continuing to reject other configuration changes. Internal checkpoint filenames such as `pipeline_config.json` remain unchanged.

Update your own terminal commands and external shortcuts to use `scripts/` and `outputs/runs/`. Archived scripts and historical logs are retained as historical artifacts; their embedded old paths are not maintained entry points.

## Relocation checks

`outputs/checks/restructure_integrity.json` records the relocation checks:

- Original annotations and all 95 raw images match their pre-move SHA-256 hashes.
- Nine saved run configuration/identity/selection JSON files match their pre-move hashes.
- The split manifest differs only in relocated image paths; fingerprint, assignments and annotation status are unchanged.
- All prepared image references resolve; counts remain 63 train, 16 validation and 16 test.
- Active Python command and package files parse successfully.

No training or automated test suite was run as part of this folder reorganization.
