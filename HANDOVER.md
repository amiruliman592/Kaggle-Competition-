# HANDOVER — Solar Filament Segmentation 2026

Written 2026-10-19 at the end of session 1 (Phase 0 + 1 "foundation"). Read this first in a new session, then `CLAUDE.md` and `Project_Phase_Plan.md`.

## 1. Goal
- Kaggle task: **instance segmentation** of solar filaments in 2048×2048 GONG H-alpha images. One mask per filament, submitted as RLE in one CSV (`filament_id,segmentation_rle`).
- Metric: Panoptic Quality (PQ = SQ × RQ, match at IoU > 0.5). Fragmenting one filament or merging two is penalised. Target PQ > 0.35.
- Deadline Nov 15, 2026 (code freeze Nov 11). Final deliverables also need a public repo, notebook, requirements.txt and a short report (30% of grade is qualitative).
- Training must run on **Google Colab** (user's local GPU is a 4 GB GTX 1650; Detectron2 is not installed locally). Claude gives the user code/instructions; the user runs them and reports back.

## 2. Current state
- **Phase 0 and 1: code done and tested locally, NOT yet run on Colab.**
- Local: 14/14 tests pass (`python -m pytest -q -p no:warnings tests`, ~85 s). GT-vs-GT on the val split gives PQ = 1.0. Frozen split created. EDA run (`output/eda/`, gitignored).
- **Nothing is committed or pushed.** Branch `Dev`, remote `origin` = github.com/amiruliman592/Kaggle-Competition-. The user was asked whether to commit/push and had not yet answered.
- Not yet verified (open Phase 0/1 exit criteria):
  - Colab dry run (`python run_pipeline.py --dry-run`) end to end with Detectron2.
  - Local PQ matches the organisers' Kaggle `self-evaluation-notebook` within ±0.001 (page only renders in a browser; Claude could not read its code).
- No real model trained, no Kaggle submission made yet.

## 3. Key findings (not obvious from the code)
- **Multi-annotator structure:** the JSON has 1,154 image entries for only **707 files**. 296 files were annotated 2–3 times independently (each set is a separate COCO "image", ids like `050103-20111116063134Lh`). Overlapping sets agree with median IoU ~0.56. Everything is keyed by file stem; `src/annotations.py` picks ONE set per file (`ANNOTATOR_POLICY="first"` → 5,251 instances, ~7.4/file). The 8,199 / ~7.1 figures in the plan counted all sets.
- **Human ceiling:** inter-annotator PQ ≈ **0.336** (SQ 0.63, RQ 0.53, Dice 0.61) over 447 comparisons. So PQ 0.35 is about human-agreement level; RQ (count/fragmentation) matters as much as outline precision.
- **Leakage:** 276 of 656 training dates have another image within 1 day. Split therefore groups by 27-day blocks.
- Test set: 180 files, no overlap with train files; 31 test dates also occur in train dates.
- Stations from filename char 15 (B Big Bear, C Cerro Tololo, L Learmonth, M Mauna Loa, T El Teide, U Udaipur), roughly balanced.
- `iscrowd` all 0; each annotation is a single polygon; areas 9–37,739 px² (raw) with only 51 "small" (<200) instances in the selected set.
- Categories: Left / Right / Unidentifiable (Ambiguous is unused after selection). Model is single-class (`NUM_CLASSES=1`).

## 4. Files that changed (all uncommitted)
New:
- `src/annotations.py` — loads COCO grouped by file stem; `select_annotations`, `load_gt_rles`.
- `src/split.py` — StratifiedGroupKFold (5 folds, fold 0 = val), 27-day groups, count-quantile strata, seed 42. Writes `splits/val_ids.json` once; `--force` to regenerate.
- `splits/val_ids.json` — **frozen: 566 train / 141 val. Never regenerate.**
- `src/validate_submission.py` — header, id format, test-image membership, bare-counts RLE, decode sample.
- `src/utils/{__init__,rle,seed,exp_log}.py` — RLE helpers, seeding, `experiments/log.csv` logger.
- `experiments/log.csv` — header only (schema from the plan).
- `tests/{conftest,test_metric,test_rle_and_submission,test_split}.py` — 14 tests.
- `notebooks/colab_foundation.ipynb` — 18-cell Colab notebook (clone → install → lock versions → tests → EDA → dry run → download results → Kaggle-notebook comparison steps).
- `HANDOVER.md` — this file.

Rewritten/modified:
- `src/config.py` — nested `CFG.paths / data / model / train / tiling / infer`; env vars `SOLAR_ROOT`, `SOLAR_DATA_DIR`, `SOLAR_OUTPUT_DIR`; no mkdir at import (call `CFG.paths.ensure_output_dirs()`).
- `src/evaluate.py` — dataset-pooled PQ/SQ/RQ, per-image mean PQ, semantic Dice, matched Dice, IoU percentiles, 1→N fragmentation and M→1 merge counters, PQ by size bin (<200, 200–2000, >2000), CLI (`--pred`, `--gt-vs-gt`).
- `src/dataset.py`, `src/train.py`, `src/inference.py`, `src/eda.py`, `src/visualize.py`, `run_pipeline.py` — updated to the new config/annotation layer; `run_pipeline.py` has `--step {eda,split,train,val,infer,all}`, `--dry-run`, `--name`, `--weights`.
- `requirements.txt` — loose pins with notes; exact cloud versions to come from `requirements-lock.txt`.
- `.gitignore` — added `submissions/*.csv`, `.pytest_cache/`.
- `CLAUDE.md` — added multi-annotator warning, human-ceiling PQ, frozen-split note, corrected instance stats.

Bugs fixed in old code: `total_mem` (should be `total_memory`) crash; inference kept all dense masks in RAM (now RLE immediately); train/test resize mismatch (old train used 640–800 multiscale, test 1024); old split was random per COCO entry (leaky); old PQ averaged per image instead of pooling.

## 5. Failed attempts / gotchas
- **WebFetch of the Kaggle self-evaluation notebook returned only the title** (JS-rendered). User must paste its PQ cell(s) and printed numbers.
- **Validator bug (fixed):** first version rejected `[` `]` in RLE strings; COCO counts legitimately contain them. Now only quotes/whitespace are forbidden (a size array is caught by the 2-column check).
- Detectron2 cannot be imported locally, so `train.py`, `inference.py`, `dataset.register_datasets`, and `run_pipeline.py --dry-run` are **untested**. IDE "missing import" errors for detectron2 are expected. Likely first failure points on Colab: Detectron2 build vs. Colab's torch/numpy versions, `T.ResizeShortestEdge(short_edge_length=tuple)` usage, DatasetCatalog re-registration, `image_id` being a string stem.
- `run_pipeline.py --dry-run` locally just prints that Detectron2 is required; that is correct behaviour.
- Windows console shows mojibake for "—" in some prints; cosmetic.
- Windows Python is 3.13 locally; Colab will differ.

## 6. Decisions made / open
- Baseline model: Detectron2 Mask R-CNN R50-FPN with anchors 8–128 (per plan). The user knows YOLOv8-seg and was asked whether to prefer it; **unanswered** (default stays Mask R-CNN).
- `ANNOTATOR_POLICY="first"`; alternatives ("most", or training on all sets) to compare in Phase 2.
- The user explicitly wants Claude to commit/push only on request; no attribution decisions pending beyond the standard co-author trailer.
- User is new to segmentation; explanations so far have been plain-language (exam analogy: train = textbook, 141 val = practice test, 180 test = real exam scored by Kaggle).

## 7. Next steps
1. Commit and push to `Dev` (ask the user; check `git status` first; dataset images are already tracked, ~670 MB pack).
2. User runs `notebooks/colab_foundation.ipynb` on a T4 (GitHub tab → branch `Dev`; `GITHUB_TOKEN` secret if repo is private). Expect "14 passed", then dry run writing `submissions/submission_dryrun.csv` and one row in `experiments/log.csv`.
3. User returns: `requirements-lock.txt`, `log.csv`, any tracebacks. **Fix whatever the dry run breaks** (most likely in the Detectron2 files, see §5).
4. User opens the Kaggle self-evaluation notebook, sends its PQ code + printed numbers; Claude compares against `src/evaluate.py` (pooled vs per-image, `> 0.5` vs `>= 0.5`, overlap handling) and adjusts until within ±0.001.
5. Freeze `requirements.txt` from the lock file. Tick Phase 0/1 gates in `Project_Phase_Plan.md`.
6. **Phase 2:** real baseline training on Colab (LR 0.0025, batch 2, 5–7k iters at 1024), val PQ via `run_pipeline.py --step val`, produce `submissions/submission_baseline_r50.csv`, validate, first Kaggle upload; record val PQ vs public LB in `experiments/log.csv`.
7. Later (per plan): CLAHE + solar-disk masking, 2048 tiling with stitching, anti-fragmentation post-processing, Mask2Former/SAM 2, TTA/ensembling.

## 8. Standing rules (from CLAUDE.md)
- Never list/read files in `train_images/` or `test_images/`; never print the 48 MB annotations JSON (parse only in Python).
- Seeds 42; submissions go to `submissions/submission_<name>.csv`; validate before upload.
