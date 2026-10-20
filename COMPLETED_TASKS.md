# Completed Tasks — Solar Filament Segmentation 2026

A running log of finished work. Newest phase at the bottom. For open work see `HANDOVER.md` §7 and `Project_Phase_Plan.md`.

## Phase 0 — Environment & tooling (done 2026-10-19)
- [x] Config rewritten as nested `CFG.paths / data / model / train / tiling / infer` with env-var overrides (`src/config.py`).
- [x] Helper modules: RLE encode/decode, seeding (42), experiment logger (`src/utils/`).
- [x] `experiments/log.csv` created with the plan's schema.
- [x] `requirements.txt` (loose pins) written.
- [x] Colab notebook `notebooks/colab_foundation.ipynb` (clone → install → lock → tests → EDA → dry run).
- [x] Colab T4 dry run passed end to end (exp001: 5 iters, plumbing only). Detectron2 built on Colab.
- [x] `requirements-lock.txt` saved from the Colab run (torch 2.11+cu130, Python 3.13, Detectron2 `1e3e13b`).
- [x] Old-code bugs fixed: `total_mem` crash, dense masks kept in RAM during inference, train/test resize mismatch.

## Phase 1 — EDA, metric & validation split (done 2026-10-19)
- [x] Multi-annotator structure found and handled: 1,154 JSON entries → 707 files; one annotation set per file (`src/annotations.py`, policy `first`, 5,251 instances).
- [x] Human ceiling measured: inter-annotator PQ ≈ 0.336 (SQ 0.63, RQ 0.53).
- [x] Leak-free split frozen: `splits/val_ids.json` (566 train / 141 val, 27-day time blocks, seed 42).
- [x] Local evaluator `src/evaluate.py`: pooled PQ/SQ/RQ, per-image PQ, Dice, fragmentation/merge counters, PQ by size bin.
- [x] Submission validator `src/validate_submission.py` (header, ids, RLE format, decode check).
- [x] Test suite: 14 tests (13/14 on Colab; sklearn-version-dependent split test replaced by `test_regenerated_split_is_valid`).
- [x] EDA run locally and on Colab; numbers match.
- [x] *(Done in Phase 2, step 2A, 2026-10-20)* Local PQ matches the official Kaggle notebook: **0.3296 vs 0.3296** on 62 multi-annotated val files (annotator 1 = GT, annotator 2 = pred), built by `src/make_metric_check_csvs.py`.

- [x] 2026-10-20 — Official metric code saved (`SelfEvaluationResult/official_metric.py`) and analysed (`SelfEvaluationResult/FINDINGS.md`): multi-annotator pooling and non-unique matching identified and measured.

- [x] 2026-10-20 — Official-metric mode in `src/evaluate.py` (all annotator sets, Kaggle matching); val PQ now logged with it. 21 tests pass.

## Phase 2 — setup (code ready, NOT yet run)
- [x] Config: anchor aspect ratios `[0.2, 0.5, 1, 2, 5]`, 6,000 iterations, checkpoint every 1,000, periodic COCO eval off.
- [x] Training resumes from the last checkpoint; checkpoints per experiment in `output/detectron2/<name>/`.
- [x] Inference uses the same anchors; each mask keeps its confidence score.
- [x] Val sweep of score threshold × minimum area (`src/sweep.py`, `run_pipeline.py --step sweep`).
- [x] `run_pipeline.py`: `--max-iter`, `--no-resume`, `--score-thr`, `--min-area`.
- [x] Colab notebook `notebooks/colab_phase2_baseline.ipynb` (Drive-backed: train → loss plot → val → sweep → submission).

## Planning & decisions
- [x] 2026-10-20 — Lecture notes (MCTA 4364, W00–W14) reviewed; optional methods added to `Project_Phase_Plan.md` (Phase 2–8 and appendix §14).
- [x] 2026-10-20 — Baseline model chosen: **Mask R-CNN R50-FPN** first, YOLOv8-seg later as a comparison.
