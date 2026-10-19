# HANDOVER — Solar Filament Segmentation 2026

Written 2026-10-19 at the end of session 1 (Phase 0 + 1 "foundation"). Read this first in a new session, then `CLAUDE.md` and `Project_Phase_Plan.md`.

## 1. Goal
- Kaggle task: **instance segmentation** of solar filaments in 2048×2048 GONG H-alpha images. One mask per filament, submitted as RLE in one CSV (`filament_id,segmentation_rle`).
- Metric: Panoptic Quality (PQ = SQ × RQ, match at IoU > 0.5). Fragmenting one filament or merging two is penalised. Target PQ > 0.35.
- Deadline Nov 15, 2026 (code freeze Nov 11). Final deliverables also need a public repo, notebook, requirements.txt and a short report (30% of grade is qualitative).
- Training must run on **Google Colab** (user's local GPU is a 4 GB GTX 1650; Detectron2 is not installed locally). Claude gives the user code/instructions; the user runs them and reports back.

## 2. Current state (updated 2026-10-19, after Colab dry run)
- **Phase 0: DONE.** Colab T4 dry run passed end to end (exp001 in `experiments/log.csv`: 5 iters, PQ 0, Dice 0.0036, 1.67 s/img — values meaningless, plumbing proven). Detectron2 built and imported on Colab. `requirements-lock.txt` saved from that run (torch 2.11+cu130, Python 3.13, Detectron2 commit `1e3e13b`).
- **Phase 1: done.** The one open gate (local PQ vs the organisers' Kaggle `self-evaluation-notebook`, ±0.001) was **moved into Phase 2 by the user** — it is Phase 2 step 2A below. Still unverified.
- **Phase 2: NOT started. The user said: do not execute Phase 2 until they ask.**
- Tests: 14/14 locally. On Colab 13/14 passed; the failure was `test_frozen_split_is_reproducible` (sklearn 1.6.1 on Colab vs 1.8.0 locally produces different folds). Replaced by `test_regenerated_split_is_valid`. The frozen file is the source of truth — never regenerate it.
- EDA on Colab matches local numbers.
- Commits are local only on `Dev` (user asked to commit locally first; push not yet requested). The user ran the notebook from a clone, so the remote apparently had the code — check `git status`/`git log origin/Dev` before assuming. Remote `origin` = github.com/amiruliman592/Kaggle-Competition-.
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
- Detectron2 cannot be imported locally (IDE "missing import" errors are expected). The Detectron2 code paths ran fine in the Colab dry run, but only for 5 iterations on 8 images; a full run is still untested (periodic COCOEvaluator with string `image_id`s, checkpointing).
- **Notebook generation bug (fixed):** escape sequences like `\n` written through the Bash heredoc came out as real newlines in the .ipynb (f-string SyntaxError in the freeze cell). When generating notebook/code text via shell, avoid backslashes (use `chr(10)`) and compile-check every cell afterwards.
- Colab ships both `opencv-python` 5.0 and `opencv-python-headless` 4.14 (same `cv2`); harmless so far.
- `run_pipeline.py --dry-run` locally just prints that Detectron2 is required; that is correct behaviour.
- Windows console shows mojibake for "—" in some prints; cosmetic.
- Windows Python is 3.13 locally; Colab will differ.

## 6. Decisions made / open
- Baseline model: Detectron2 Mask R-CNN R50-FPN with anchors 8–128 (per plan). The user knows YOLOv8-seg and was asked whether to prefer it; **unanswered** (default stays Mask R-CNN).
- `ANNOTATOR_POLICY="first"`; alternatives ("most", or training on all sets) to compare in Phase 2.
- The user explicitly wants Claude to commit/push only on request; no attribution decisions pending beyond the standard co-author trailer.
- User is new to segmentation; explanations so far have been plain-language (exam analogy: train = textbook, 141 val = practice test, 180 test = real exam scored by Kaggle).

## 7. Next steps
1. ~~Commit~~ (done, local). Push `Dev` when the user asks.
2. ~~Colab dry run~~ (done, exp001). ~~Lock file~~ (saved). Phase 0 and the split gate are ticked in `Project_Phase_Plan.md`.
3. **Phase 2 — Baseline model, metric alignment & first Kaggle submission (WAITING for user go-ahead; do not start on your own).**
   - **2-0 Decide model** with the user: Mask R-CNN R50-FPN (default, already proven by the dry run) vs YOLOv8-seg (user knows it). Recommendation given: Mask R-CNN first, YOLOv8-seg later as a comparison.
   - **2A Kaggle self-check / metric alignment (carried over from Phase 1).**
     1. User opens https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook → Copy & Edit, runs it unchanged, records which GT/prediction files it reads and the PQ/SQ/RQ/Dice it prints.
     2. User pastes its PQ/matching cell(s) + printed numbers to Claude.
     3. Claude compares against `src/evaluate.py`: pooled vs per-image PQ, `IoU > 0.5` vs `>= 0.5`, matching of overlapping predictions, which Dice it uses (torchmetrics DiceScore settings), how it treats multi-annotator GT. Adjust `evaluate.py` + tests until the same inputs give the same PQ within ±0.001.
     4. If possible, run the official notebook on our own validation predictions (CSV from 2C) and compare numbers directly. Gate: tick the ±0.001 checkbox (Phase 1 section) in `Project_Phase_Plan.md`.
     - 2A can run in parallel with 2B; it must be done before trusting val PQ for model selection.
   - **2B Real baseline training on Colab:** Claude writes a Phase 2 Colab notebook (e.g. `notebooks/colab_phase2_baseline.ipynb`). Per plan: R50-FPN, anchors 8–128, aspect ratios `[0.2, 0.5, 1.0, 2.0, 5.0]` (not yet in config), 1024 input, LR 0.0025, batch 2, 5–7k iters, flips + brightness/contrast. Save checkpoints to Google Drive (Colab sessions die). Untested so far: periodic COCOEvaluator with string `image_id`s — consider setting `EVAL_PERIOD=0` and using our PQ instead.
   - **2C Validation:** `python run_pipeline.py --step val --weights <ckpt> --name baseline_r50` → val PQ/SQ/RQ, frag/merge, size-bin PQ; logged to `experiments/log.csv`. Consider a quick sweep of `SCORE_THRESHOLD` / `MIN_AREA` on val.
   - **2D Submission:** `python run_pipeline.py --step infer --weights <ckpt> --name baseline_r50` → `submissions/submission_baseline_r50.csv`, passes `validate_submission.py` (all 180 test images), user uploads to Kaggle, records public LB score in `lb_score` of the log row.
   - **Phase 2 exit gates:** baseline trained + evaluated on frozen val; first valid Kaggle submission scored with 0 errors; local metric aligned with the official notebook (2A).
4. Later (per plan): CLAHE + solar-disk masking, 2048 tiling with stitching, anti-fragmentation post-processing, Mask2Former/SAM 2, TTA/ensembling.

## 8. Standing rules (from CLAUDE.md)
- Never list/read files in `train_images/` or `test_images/`; never print the 48 MB annotations JSON (parse only in Python).
- Seeds 42; submissions go to `submissions/submission_<name>.csv`; validate before upload.
