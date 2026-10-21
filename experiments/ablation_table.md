# Ablation Table v1 — Phase 3 (frozen val, 141 images / 233 annotator entries)

All numbers: official Kaggle PQ (`evaluate_official`) and Dice (`official_dice`, samplewise = torchmetrics
`DiceScore` default). Model: Mask R-CNN R50-FPN baseline (exp003 checkpoint), raw predictions saved once
(`Result Phase 3/Result Step 1/raw_preds_baseline_r50_val.json.gz`), post-processing applied offline
(`src/ablate_postprocess.py`, raw rows in `Result Phase 3/Result Step 1/ablation_postproc_round{1,2}.json`).

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

## Retraining (step 3, 2026-10-20) — same recipe as B0 (R50-FPN, 6,000 it), one change per run
Each model scored over thr {0.5…0.85} × overlap rule {carve, drop 0.2}; rows show each model's best and the
plateau (raw rows: `Result Phase 3/Result Step 3/eval_step3.json`).

| # | Run | Change | Best official PQ | Δ vs B0'' | Dice | Δ Dice | Plateau (thr 0.6–0.7, drop 0.2) | Verdict |
|---|---|---|---|---|---|---|---|---|
| B0'' | baseline_r50 | post-processing row 5 | 0.3899 (thr 0.65) | — | 0.6349 | — | 0.388–0.390 | reference |
| A | r50_clahe_disk | disk masking + CLAHE (train + predict) | 0.3901 (thr 0.6) | +0.0002 | 0.6413 | +0.006 | 0.387–0.390 | **no PQ gain** → not adopted |
| B | r50_allann | every annotator set = training sample (921 vs 566) | **0.3979** (thr 0.6) | **+0.0080** | **0.6426** | **+0.008** | **0.396–0.398** | **adopted** |

Run B notes: higher RQ (0.608 vs 0.597; FP 617 vs 664 at similar TP), fewer fragmentations (18 vs 23 at
thr 0.6), slightly more merges (31 vs 26). Small filaments (< 200 px²): still 0/10 for every model → Phase 4.
Submission built from Run B raw test predictions: `submissions/submission_r50_allann_pp.csv` (thr 0.6, drop 0.2,
1,279 masks, passes validator). **Public LB 0.35** (first gain over 0.34).

**Phase 3 status:** best val official PQ 0.3979 (B0 + 0.0074); gate B0 + 0.030 = 0.4205 not met.

## Phase 4 — native resolution 2048 (2026-10-20, Colab A100 80 GB)
`r50_allann` recipe (all-annotator labels, 6,000 it, batch 2, anchors 8–128), only resolution / backbone changed.
Scored like step 3: closing 5, `MIN_AREA` 50, overlap drop 0.2, thr {0.5…0.9} (raw rows:
`Result Phase 4/eval_phase4.json`). Sanity probe: X101 at 2048 peaks at 20.2 GB GPU memory; prediction 0.5–0.7 s/img.

| # | Run | Change | Best official PQ | Δ vs B | Dice | Plateau (thr 0.6–0.8) | Small TP/FP/FN | PQ medium / large | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| B | r50_allann | 1024 (reference) | 0.3979 (thr 0.6) | — | 0.6426 | 0.390–0.398 | 0/0/10 | 0.372 / 0.434 | reference |
| C | r50_2048 | resolution 1024 → 2048 | 0.4250 (thr 0.7) | **+0.027** | 0.6380 (0.6484 at thr 0.6, PQ 0.4231) | 0.415–0.425 | 0/0/10 | 0.397 / 0.471 | **adopted** |
| D | x101_2048 | + backbone R50 → X101 | **0.4264** (thr 0.6) | **+0.029** | 0.6417 | **0.426–0.426** | 2/1/8 (thr 0.6) | 0.406 / 0.467 | **best**, flattest plateau |

Notes: the gain comes from both size bins (medium +0.03, large +0.04) and from SQ (+0.013) and RQ (+0.03), not
from small filaments, which are still almost all missed (X101 finds 2/10 at thr ≤ 0.6). Large-bin PQ rose, so the
conditional bigger-anchor run is **not needed**. X101 vs R50 at 2048 is within noise on PQ (+0.0014) but has a wider
plateau (thr 0.6–0.8 all 0.426) and fewer merges (25 vs 28 at thr 0.6). Phase 3 gate (0.4205) is now passed.
Submissions (thr 0.6, drop 0.2, pass validator): `submission_x101_2048_pp.csv` (1,317 masks) → **public LB 0.37**
(best so far); `submission_r50_2048_pp.csv` (1,313 masks) → **public LB 0.36**. LB gain over B: +0.02 (val +0.026–0.029);
ranking D > C kept on LB. Val − LB gap 0.056 / 0.063 (earlier runs 0.047–0.050; LB shown to 2 dp, ±0.005).
