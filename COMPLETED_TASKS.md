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

## Phase 2 — baseline run (done on Colab 2026-10-20; files in `Result Phase 2/`)
- [x] Trained Mask R-CNN R50-FPN, 6,000 iters on T4 (~1.3 h). Loss fell from 1.43 to ~0.4 early on. "No CUDA drivers" log line is TensorFlow, harmless.
- [x] Val, default thresholds (score 0.5, min area 50) — exp002: **official PQ 0.3776** (SQ 0.651, RQ 0.580); first-annotator PQ 0.3761; frag 66, merge 37.
- [x] Sweep (30 cells, `sweep_baseline_r50.json`): best **score 0.8**, official PQ 0.3905. Scores 0.6–0.8 are within 0.0006 (flat). **Min area has no effect** (no predicted mask is < 200 px²). 0.8 is the edge of the grid, so 0.85–0.9 is still untested.
- [x] Val at score 0.8 — exp003: **official PQ 0.3905** (SQ 0.656, RQ 0.596; TP 903 / FP 448 / FN 778); frag 23, merge 26; 0.42 s/img. Already above the 0.35 target and the 0.336 inter-annotator ceiling.
- [x] Size bins (first annotator): small 0/0/10 (all missed), medium PQ 0.365, large PQ 0.449.
- [x] Submission `submission_baseline_r50.csv` (score 0.8): 1,077 rows over 176/180 images, passes the validator. 4 images have no masks; this is a warning, not an error (official metric counts them as FN). Validator changed accordingly.
- [x] First Kaggle upload REJECTED: overlapping masks not allowed (18 images). Fixed in code (`remove_overlaps`, validator check) and the CSV repaired locally → `submissions/submission_baseline_r50_nooverlap.csv` (1,072 masks, passes validator). See `SelfEvaluationResult/FINDINGS.md` §3b.
- [x] Uploaded `submission_baseline_r50_nooverlap.csv` → **public LB 0.34** (shown to 2 decimals; ~50% of test). Recorded as `lb_score` of exp003. Val 0.3905 vs LB 0.34: gap ~0.05, cause not yet known (see HANDOVER).
- [x] 2026-10-20 — Competition rules read from `Solar Filament Overview.txt` and recorded in `HANDOVER.md` §2b (deliverables, 70/30 rubric with Dice, ~50% public LB). Dice alignment added as Phase 3 step 0; repo/reproducibility rules added to Phase 8.

- [x] **Phase 2 complete (2026-10-20).** All required gates done; optional classical baseline skipped.
- [x] Leaderboard reviewed: top 0.60–0.64 judged not comparable (above human ceiling, organisers' hacking warning); same-day label copying and external MAGFiLO labels ruled out. Kaggle limit: 5 uploads/day, 3 final picks.

## Planning & decisions
- [x] 2026-10-20 — Lecture notes (MCTA 4364, W00–W14) reviewed; optional methods added to `Project_Phase_Plan.md` (Phase 2–8 and appendix §14).
- [x] 2026-10-20 — Baseline model chosen: **Mask R-CNN R50-FPN** first, YOLOv8-seg later as a comparison.
