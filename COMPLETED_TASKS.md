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

## Phase 3 (started 2026-10-20)
- [x] Step 1 code: `run_pipeline.py --step dump` (raw val+test masks, score ≥ 0.2) + `notebooks/colab_phase3_dump_preds.ipynb`; `src/postprocess.py` (CPU post-processing). Colab run in progress.
- [x] Step 0 Dice alignment: `evaluate.official_dice` (samplewise/global torchmetrics Dice, notebook pair Dice/IoU means) — matches torchmetrics exactly and the Kaggle plots (0.665 / 0.519). `val_dice` in the log is now `official_Dice_samplewise`. Details: `SelfEvaluationResult/FINDINGS.md` §3c.

- [x] Step 1 Colab dump done: 1,813 raw val masks / 2,356 raw test masks (score ≥ 0.2) in `Result Phase 3/`.
- [x] Step 2 post-processing ablations (local CPU) → `experiments/ablation_table.md`. Overlap removal costs −0.004 PQ (0.3905 → 0.3865, explains part of val–LB gap). Closing kernel, disk filter, threshold > 0.8: no gain. New overlap rule (drop mask if > 20% covered) + thr 0.65: **val PQ 0.3899, Dice 0.6349** (+0.0034 / +0.0155) — now CFG defaults. Shape filter not run (post-processing has no headroom left).
- [x] `src/submit_from_raw.py`: submission CSV from raw test predictions on CPU → `submissions/submission_baseline_r50_pp.csv` (1,312 masks, 177/180 images, passes validator). Uploaded: **LB 0.34** (exp004).
- [x] Step 3 code: `src/preprocess.py` (disk mask + CLAHE, cached once per run; `image_path` used by dataset/val/dump/infer), `TRAIN_LABEL_POLICY` "all" (921 samples / 6,521 instances vs 566 / 4,221), `run_pipeline.py --step preprocess --preproc --label-policy`, notebook `notebooks/colab_phase3_retrain.ipynb` (Run A `r50_clahe_disk`, Run B `r50_allann`, each saves raw val/test predictions). 29 tests pass. Colab runs not started.
- [x] Step 3 retraining scored (exp005, exp006; `experiments/ablation_table.md`): Run A disk+CLAHE **no PQ gain** (0.3901); Run B all-annotator labels **val PQ 0.3979, Dice 0.6426** (+0.008 / +0.008) → adopted. `submissions/submission_r50_allann_pp.csv` uploaded: **LB 0.35** (best so far). Result files moved by user to `Result Phase 3/Result Step 1|3/`.
- [x] One-window results viewer `src/viewer.py` (GT left, predictions right, TP green / FP red / FN red fill; Prev/Next, model switch, worst-PQ sort, Save PNG). `close_all` closing now on padded crops: identical masks, ~7x faster. 31 tests pass.

## Phase 4 (started 2026-10-20)
- [x] Code ready: `run_pipeline.py --backbone r50|x101 --resolution N` (sets train + infer size together; `MODEL_CONFIGS` in `src/config.py`); data-loader workers follow the CPU count; training prints peak GPU memory. Notebook `notebooks/colab_phase4_native.ipynb` (A100: sanity probe → `r50_2048` → `x101_2048`, each dumps raw val/test predictions to `MyDrive/solar_filament/Phase 4/`; stops on any failed command). `tests/test_pipeline_flags.py` (+3); 34 tests pass.
- [x] Colab A100 run done (raw preds in `Result Phase 4/`). Sanity probe: X101 at 2048 peaks at 20.2 GB, 0.5–0.7 s/img prediction.
- [x] Scored locally (exp007, exp008; `experiments/ablation_table.md`, `Result Phase 4/eval_phase4.json`): **r50_2048 val PQ 0.4250** (thr 0.7) / 0.4231 Dice 0.6484 (thr 0.6); **x101_2048 val PQ 0.4264, Dice 0.6417** (thr 0.6, plateau 0.426 over thr 0.6–0.8). +0.029 vs r50_allann; medium and large bins both up; small filaments still mostly missed (X101 2/10). Bigger-anchor run not needed (large bin improved). Phase 3 gate 0.4205 now passed.
- [x] Submissions built (thr 0.6, drop 0.2, pass validator): `submission_x101_2048_pp.csv` (1,317 masks), `submission_r50_2048_pp.csv` (1,313).
- [x] 2026-10-21 — Uploaded: **x101_2048 LB 0.37** (best so far, exp008), **r50_2048 LB 0.36** (exp007). Val gain carried over (+0.02 LB vs +0.03 val, same ranking); val − LB gap 0.056–0.063, slightly wider than before (0.047–0.050) but within 2-dp rounding.

## Phase 5 (started 2026-10-21)
- [x] Step 1 code: `src/sam_refine.py` — SAM 2.1 (hiera-large) redraws every `x101_2048` raw mask (score ≥ 0.5) from a square crop around its box; prompts `box` / `boxpts` (box + 3 points along the long axis); `dump` (GPU) → `ablate` (CPU: modes sam / union / inter / gate × thr 0.5–0.8, official PQ + Dice) → `export` (normal raw file for `submit_from_raw.py`). The `orig` rows reproduce exp008 exactly (0.4264 / 0.6417 at thr 0.6). Notebook `notebooks/colab_phase5_sam2.ipynb` (T4). `tests/test_sam_refine.py` (+4).
- [x] Step 1 run (Colab T4, files in `Result Step 5/Step 1/`) and scored (exp009, `experiments/ablation_table.md`): **SAM 2 rejected** — every prompt × combination is below `x101_2048` (best: box union 0.4076 vs 0.4264; Dice ≤ 0.62 vs 0.6417). SAM masks are ~15% tighter than the annotators' outlines.
- [x] Step 1b CPU ensemble (`src/ensemble.py`, `tests/test_ensemble.py` +3; exp010): x101_2048 + r50_2048 fused, IoU 0.5, union, thr 0.6 → **val PQ 0.4366, Dice 0.6505** (+0.010 / +0.009, new best). Dilation 1–2 px hurts. `submissions/submission_ens_x101_r50_2048_pp.csv` built (1,278 masks, passes validator) → **public LB 0.38** (best so far, 2026-10-21).
- [x] Step 2 code: `src/m2f.py` (HF `transformers` 5.17 Mask2Former, Swin-S COCO-instance weights, 1 class; 1024 crops of native 2048, flips + brightness/contrast, AdamW 1e-4 / backbone 1e-5, poly LR, bf16 on A100, resumable `last.pt`; full-2048 prediction → standard raw dump). Smoke-tested locally (20 its at 256 px on GTX 1650, dump of 2 images; weights fetched with curl because Avast breaks Python HTTPS locally). Finer Mask R-CNN mask head: `CFG.model.MASK_POOLER_RES` / `--mask-res 28` (train + inference). Notebook `notebooks/colab_phase5_a100.ipynb` (probes → `m2f_swin_s` capped at ~4 h from measured speed → `x101_2048_m28`). 43 tests pass. Colab run not started.

## Planning & decisions
- [x] 2026-10-20 — Lecture notes (MCTA 4364, W00–W14) reviewed; optional methods added to `Project_Phase_Plan.md` (Phase 2–8 and appendix §14).
- [x] 2026-10-20 — Baseline model chosen: **Mask R-CNN R50-FPN** first, YOLOv8-seg later as a comparison.
- [x] 2026-10-20 — Phase 4 plan agreed (not started): full 2048 image on A100 80 GB (no tiles); one notebook runs R50 then X101 at 2048 back to back; bigger anchors only if the large bin drops (decided after results). ~40–45 of ~188 units.
