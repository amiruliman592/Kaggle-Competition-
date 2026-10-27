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

## Phase 5 step 1 — SAM 2 box refinement of x101_2048 (2026-10-21, Colab T4, no training)
SAM 2.1 hiera-large redraws each `x101_2048` mask (score ≥ 0.5) from a square crop around its box
(`src/sam_refine.py`; rows: `Result Step 5/Step 1/sam_ablation_x101_2048.json`). Close 5, min area 50, drop 0.2.

| Prompt | Combination | Best official PQ (thr) | Δ vs D (0.4264) | Dice at that thr |
|---|---|---|---|---|
| — | original (D) | **0.4264** (0.6) | — | **0.6417** |
| box | SAM replaces | 0.3577 (0.8) | −0.069 | 0.5526 |
| box | union | 0.4076 (0.8) | −0.019 | 0.5857 |
| box | intersection | 0.3683 (0.8) | −0.058 | 0.5782 |
| box | gate (SAM if IoU ≥ 0.5) | 0.3831 (0.8) | −0.043 | 0.5853 |
| box+3 pts | SAM replaces | 0.3653 (0.8) | −0.061 | 0.5550 |
| box+3 pts | union | 0.4028 (0.8) | −0.024 | 0.5842 |
| box+3 pts | intersection | 0.3804 (0.8) | −0.046 | 0.5828 |
| box+3 pts | gate | 0.3884 (0.8) | −0.038 | 0.5840 |

**Verdict: rejected** (every variant loses PQ and Dice). SAM masks are a median 15% smaller than ours (it traces the
dark core; annotators draw a looser outline) and ~10% bleed out to 1.4–1.5× the area; median IoU with our masks is
only 0.74, so even the gate loses TPs. Matches the plan's caveat: the labels, not mask sharpness, set SQ.

## Phase 5 step 1b — CPU ensemble of x101_2048 + r50_2048 (2026-10-21, 0 GPU units)
`src/ensemble.py`: masks of the two models with IoU ≥ match IoU are one filament; score = mean over both models
(a model that missed it adds 0); mask = best member / union / intersection. Close 5, min area 50, drop 0.2
(rows: `Result Step 5/Step 1b/ensemble_ablation_x101_r50_2048.json`).

| Fusion | thr 0.5 | thr 0.6 | thr 0.7 | Dice at thr 0.6 |
|---|---|---|---|---|
| x101_2048 alone (D) | 0.4190 | 0.4264 | 0.4263 | 0.6417 |
| IoU 0.5, best mask | 0.4214 | 0.4285 | 0.4298 | 0.6475 |
| **IoU 0.5, union** | 0.4301 | **0.4366** | 0.4376 | **0.6505** |
| IoU 0.5, intersection | 0.4120 | 0.4223 | 0.4251 | 0.6423 |
| IoU 0.3, union | 0.4290 | 0.4357 | 0.4372 | 0.6518 |

**Verdict: adopted** — union at thr 0.6: **PQ 0.4366 (+0.010), Dice 0.6505 (+0.009)**, RQ 0.639 → 0.652. Not a
mask-size effect: dilating masks by 1 / 2 px instead *loses* PQ (x101: 0.4196 / 0.3571; ensemble: 0.4171 / 0.3424).
Submission `submission_ens_x101_r50_2048_pp.csv` (1,278 masks, 177/180 images, passes validator) → **public LB 0.38** (best; +0.01 over x101 alone, as on val).

## Phase 5 step 2 — Mask2Former and finer mask head (2026-10-21, A100 run, scoring on CPU)
Raw files in `Result Step 5/Step 2/`. Close 5, min area 50, drop 0.2; official val PQ (Dice in brackets).

| Model alone | thr 0.5 | thr 0.6 | thr 0.7 | thr 0.8 | Frag / merge |
|---|---|---|---|---|---|
| x101_2048 (exp008) | 0.4190 | 0.4264 (0.6417) | 0.4263 | – | – |
| **x101_2048_m28** (56×56 mask head) | 0.4279 | 0.4355 (0.6496) | **0.4385** (0.6408) | 0.4359 | 10 / 32 |
| m2f_swin_s (Mask2Former) | 0.3676 | 0.3799 | 0.3942 | 0.4020 (0.6597) | 17 / 67 |

| Ensemble (best setting) | PQ | Dice |
|---|---|---|
| x101 + r50 (exp010) | 0.4366 | 0.6505 |
| x101 + r50 + m28 (IoU 0.5 union thr 0.6) | 0.4375 | 0.6508 |
| x101 + r50 + m2f (IoU 0.3 best thr 0.6) | 0.4256 | 0.6504 |
| all 4 (IoU 0.5 best thr 0.6) | 0.4318 | 0.6463 |
| **m28 + r50 (IoU 0.5 union thr 0.6)** | **0.4404** | **0.6501** |
| m28 + r50 (IoU 0.5 union thr 0.7) | 0.4407 | 0.6353 |

**Verdict:** the finer mask head is a clear gain (+0.012 alone, best single model); m28 + r50 is the new best
(+0.004 over exp010; thr 0.6 kept for Dice). Mask2Former is weaker alone (−0.024, twice the merges) and lowers
every ensemble it joins, despite the best Dice. **Public LB: both 0.37** (exp010 x101 + r50 stays best at 0.38): val gains this small do not show on the LB.
Submissions: `submission_ens_m28_r50_2048_pp.csv` (1,277 masks), `submission_x101_2048_m28_pp.csv` (1,176); both pass the validator.

## Phase 6 M1a — noise check, paired bootstrap (2026-10-27, Mirul, CPU)
`src/bootstrap.py`, 2,000 re-draws of the 141 val images (seed 42); "LB" = 90 images drawn with replacement, rounded
to 2 dp. Same draws for both runs of a pair. Min real gain = 1.96 × SD of the paired difference.
Output: `Result Phase 6/Mirul/bootstrap.json`. Full-val PQ reproduces `log.csv` exactly.

| B vs A | ΔPQ val | 95% CI | P(B>A) | min real gain val | min real gain LB | sim. LB B higher / tie / lower |
|---|---|---|---|---|---|---|
| exp010 vs exp006 | +0.0386 | [+0.027, +0.050] | 1.00 | 0.011 | 0.015 | 1.00 / 0.00 / 0.00 |
| exp012 vs exp010 | +0.0019 | [−0.007, +0.010] | 0.66 | 0.009 | 0.011 | 0.31 / 0.55 / 0.14 |
| exp013 vs exp010 | +0.0038 | [−0.002, +0.009] | 0.92 | 0.005 | 0.007 | 0.41 / 0.57 / 0.02 |
| exp013 vs exp012 | +0.0020 | [−0.006, +0.010] | 0.69 | 0.008 | 0.010 | 0.31 / 0.59 / 0.11 |

Single run: SD of val PQ ≈ 0.011, of a 90-image LB ≈ 0.013–0.015.
**Verdict:** a val gain must be **≥ ~0.010 PQ** (paired, same 141 images) to be real; on the LB that is ≥ ~0.015,
i.e. in practice **+0.02 on the 2-dp LB**. None of the Phase 5 step 2 gains (+0.002–0.004) clear it. exp013's real
LB drop (0.37 vs 0.38) had only ~2% chance from sampling noise alone: val and test may differ a little, so treat val
gains near the bar with extra caution. Training noise (Syaz S1) still to be added.

## Phase 6 S1 + S2 — seed repeat and 3× schedule, R50 at 1024 (2026-10-27, Syaz; T4 training, CPU scoring)
exp006 recipe (R50-FPN, 1024, all-annotator labels, batch 2, LR 0.0025), one change per run. Scored like exp006:
close 5, min area 50, overlap drop 0.2, thr 0.5–0.85, official PQ + Dice (samplewise); exp006 reproduces 0.3979 /
0.6426. Paired bootstrap = `src/bootstrap.py` (2,000 draws, seed 42), each run at its best thr. Outputs:
`Result Phase 6/Syaz/eval_s1_s2.json`, `Result Phase 6/Syaz/bootstrap_s1_s2.json`.

| Run | Change | PQ thr 0.6 | Dice thr 0.6 | best PQ (thr) | Dice at best | ΔPQ vs exp006 (95% CI) | P(B>A) | Verdict |
|---|---|---|---|---|---|---|---|---|
| exp006 | reference (seed 42, 6,000 its) | 0.3979 | 0.6426 | 0.3979 (0.6) | 0.6426 | — | — | reference |
| **exp201** S1 `syaz_r50_seed2` | `--seed 7` | 0.3979 | 0.6382 | 0.3985 (0.65) | 0.6385 | +0.0005 [−0.006, +0.007] | 0.56 | **training noise** |
| **exp202** S2 `syaz_r50_3x` | `--max-iter 18000` (LR drops 10,800 / 15,300) | 0.3907 | 0.6321 | 0.3989 (0.75) | 0.6151 | +0.0007 [−0.010, +0.011] | 0.55 | **rejected** |

**Training noise (S1):** re-training with another seed moves full-val PQ by only ~0.001 (identical 0.3979 at thr
0.6), but the PQ-vs-thr curve shifts (best thr 0.6 → 0.65) and the paired per-image difference has a 95% bar of
±0.007. So the M1 bar (**≥ ~0.010 val PQ**) stays the binding one; S1 does not raise it.
**3× schedule (S2):** no PQ gain (+0.0007, inside the noise), lower Dice at every thr ≥ 0.6, and the scores become
more confident (best thr moves to 0.75; at thr 0.6 it has more FPs, 652 vs 617). Keep **6,000 its** for S3–S5 (also
3× cheaper on a T4). S2 was resumed once from `model_0011999.pth` with optimizer + LR-scheduler state (only the data
order after the resume differs from an unbroken run).

## Phase 6 M1b — error budget of exp010 (2026-10-27, Mirul, CPU)
`src/error_budget.py`, official matching (233 annotator entries, IoU > 0.5), thr 0.6. Output:
`Result Phase 6/Mirul/error_budget_exp010.json`. PQ 0.4366, SQ 0.6695, TP 1067 / FP 527 / FN 611.
"PQ if fixed" = that box alone made perfect (FN → hit at the mean IoU, FP removed): an upper bound, not a forecast.

| Box | n | PQ if fixed | gain | Meaning |
|---|---|---|---|---|
| fn_low_score | 246 | 0.4997 | +0.063 | raw hit exists, score < 0.6 → confidence |
| fp_nobody | 341 | 0.4874 | +0.051 | real false alarm |
| fn_never | 181 | 0.4839 | +0.047 | recall (nothing within IoU 0.3) |
| fn_near_miss | 146 | 0.4752 | +0.039 | kept mask at IoU 0.3–0.5 → outline |
| fp_other_ann | 186 | 0.4630 | +0.026 | another annotator drew it → label noise, not fixable |
| fn_postproc | 38 | 0.4470 | +0.010 | good raw hit removed by min area / overlap rule |
| SQ loss (all hits IoU 1) | 1067 | 0.6522 | +0.216 | not reachable: our SQ 0.67 is already above the human 0.63 |

By size: small 12 GT entries, all missed (≤ 0.005 PQ, ignore); medium PQ 0.419, large 0.482.
By station (val ~20–28 images each, so differences of ±0.02 are noise): B 0.459, U 0.442, L 0.441, M 0.436,
C 0.428, T 0.415.
**Verdict:** the biggest box is **confidence** (246 true filaments found but scored < 0.6) followed by real false
alarms (341): the model ranks real vs fake filaments poorly at the cut-off. This is what flip TTA (M2) attacks
(agreement across views raises real ones, lowers one-off ones). Recall (181 never found) is the target of more data
(M3/S5) and the recipe (S2–S4). 35% of FPs are drawn by another annotator: label noise sets a hard floor.

## Phase 6 M2 — flip TTA of the 2048 models (2026-10-27, Mirul; Colab T4 inference, scoring on CPU)
Kimee's `x101_2048_m28` and `r50_2048` on val + test flipped h / v / hv (masks flipped back; median best IoU vs
un-flipped 0.85–0.89; r50 peak GPU 2.4 GB, 1.17 s/img on T4). Views fused with `src/ensemble.py` (score = mean
over views). Logs + grids: `Result Phase 6/Mirul/log_tta*.txt`, `tta*.json`.

| Run (val, best setting of the 30-row grid) | PQ | Dice | vs |
|---|---|---|---|
| m28, 4 views (IoU 0.5 union thr 0.7) | 0.4416 | 0.6379 | m28 alone 0.4385: +0.003 |
| r50, 4 views (IoU 0.5 union thr 0.6) | 0.4345 | 0.6517 | r50 alone 0.4231 (thr 0.6): +0.011 |
| 8 views in one fusion (IoU 0.5 best thr 0.7) | 0.4382 | 0.6369 | exp013 0.4404: −0.002 |
| **two-stage: per model 4 views (IoU 0.5 union), then 2 models (IoU 0.3 inter) thr 0.6** | **0.4480** | **0.6520** | **exp010 +0.0116, exp013 +0.0078** |
| same, thr 0.7 | 0.4461 | 0.6425 | exp010 +0.0095 |

Bootstrap (`Result Phase 6/Mirul/bootstrap_m2.json`): vs exp010 +0.0116 [+0.004, +0.020], P(better) 1.00, min real
gain for this pair 0.008; simulated LB higher / tie / lower 0.86 / 0.14 / 0.00. vs exp013 +0.0078 [+0.001, +0.014].
Caveat: setting picked from 30 on val (some optimism); the inter fusion at IoU 0.5 and the single 8-view fusion do
not show the gain, so it depends on the two-stage order.
**Verdict: adopted.** Submission `submissions/submission_tta2_m28_r50_pp.csv` (1,256 masks, validator PASS):
**public LB 0.39** (exp010 0.38) — first LB gain since Phase 5, new best (exp101).

## Phase 6 M3 — external GONG images: source check + pseudo-labeller choice (2026-10-27, Mirul; CPU)
**Source:** NSO GONG archive product `hag` (`https://gong2.nso.edu/HA/hag/YYYYMM/YYYYMMDD/<stem>.jpg`) has the same
frames and names as the competition images. 3 train files (2011 C, 2014 B, 2022 T) compared: same 2048×2048 frame,
correlation ≥ 0.99, mean |diff| 2.1–2.8 grey levels on the disk, brightness fit B = 1.00·G + 0.6–1.1. The archive copy
is RGB (channels equal) at JPEG quality ~75; the competition copy is greyscale at ~90. The Colab run measures the
effect on val PQ (archive copies of the 141 val images vs 0.4355 on the competition copies, m28 thr 0.6).
**Exclusion:** every day of a val 27-day block + every test day ±1 → 3,310 allowed days in 2011–2022.

Pseudo-label precision on val (kept mask matched by ≥ 1 annotator set at IoU > 0.5; recall averaged over sets;
usual post-processing; scratch script, CPU):

| Labeller | thr | masks/img | precision | recall | matched IoU |
|---|---|---|---|---|---|
| m28, 1 view | 0.80 | 5.38 | 0.817 | 0.570 | 0.689 |
| **m28, 1 view** | **0.85** | **4.73** | **0.840** | **0.529** | **0.694** |
| m28, 1 view | 0.90 | 4.03 | 0.864 | 0.473 | 0.696 |
| m28, 4 flips | 0.85 | 4.48 | 0.854 | 0.511 | 0.692 |
| m28 + r50, 8 views (exp101) | 0.80 | 4.87 | 0.860 | 0.547 | 0.694 |
| m28 + r50, 8 views (exp101) | 0.85 | 4.24 | 0.885 | 0.496 | 0.697 |

**Choice: m28 alone, thr 0.85.** The 8-view TTA is ~2 points more precise at equal recall but needs 8× the T4 time
(~14 h vs < 2 h for 3,000 images). For S5: about half of the filaments in a pseudo-labelled image are not labelled,
so unlabelled pixels are not reliable background; raw masks (score ≥ 0.2) are kept so S5 can ignore unsure regions.
