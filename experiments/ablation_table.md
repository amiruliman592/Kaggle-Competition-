# Ablation Table v1 — Phase 3 (frozen val, 141 images / 233 annotator entries)

All numbers: official Kaggle PQ (`evaluate_official`) and Dice (`official_dice`, samplewise = torchmetrics
`DiceScore` default). Model: Mask R-CNN R50-FPN baseline (exp003 checkpoint), raw predictions saved once
(`Result Phase 3/raw_preds_baseline_r50_val.json.gz`), post-processing applied offline
(`src/ablate_postprocess.py`, raw rows in `Result Phase 3/ablation_postproc_round{1,2}.json`).

## Post-processing (step 2, 2026-10-20)
| # | Setting | Official PQ | Δ PQ | Dice | Δ Dice | Verdict |
|---|---|---|---|---|---|---|
| B0 | Phase 2 baseline: thr 0.8, closing 5, overlaps allowed (exp003) | 0.3905 | — | 0.6088* | — | not submittable (Kaggle rejects overlaps) |
| B0' | + overlap removal (carve, higher score wins) — what was uploaded (LB 0.34) | **0.3865** | −0.0040 | 0.6194 | | new reference |
| 1 | Closing kernel 0 / 3 / 5 / 7 (thr 0.8) | 0.3864–0.3865 | 0.0000 | 0.6193–0.6195 | 0.000 | no effect → keep 5 |
| 2 | Score threshold 0.85 / 0.9 (vs 0.8) | 0.3806 / 0.3572 | −0.006 / −0.029 | 0.608 / 0.551 | ↓ | rejected |
| 3 | Score threshold 0.6 / 0.7 | 0.3744 / 0.3817 | ↓ | 0.639 / 0.633 | ↑ | PQ/Dice trade-off |
| 4 | Disk-perimeter filter (centroid > 0.95–1.0 R) | identical | 0.0000 | identical | 0.000 | no effect (no predictions off-disk) → dropped |
| 5 | Overlap rule: drop a mask whole if > 20% is covered (thr 0.65) | **0.3899** | **+0.0034** | **0.6349** | **+0.0155** | **adopted** (plateau 0.389–0.390 for drop 0.1–0.3, thr 0.65–0.75) |
| — | `MIN_AREA` 0–400 | — | 0 | — | 0 | settled in Phase 2: no predicted mask < 200 px² |

\* exp003 logged the single-annotator semantic Dice; official samplewise Dice was introduced in Phase 3.

**Current best (post-processing only): val official PQ 0.3899, Dice 0.6349** — `SCORE_THRESHOLD=0.65`,
`OVERLAP_DROP_FRAC=0.2`, closing 5, `MIN_AREA=50` (now the defaults in `src/config.py`).
Uploaded as `submission_baseline_r50_pp.csv` (exp004): **public LB 0.34**, same as B0' — expected, the +0.003 PQ gain
is below the leaderboard's 2-decimal resolution.
Phase 3 gate (PQ ≥ B0 + 0.030 = 0.4205) is **not** reachable with post-processing; needs retraining (step 3).

## Retraining (step 3)
_Pending._
