# Official metric — findings (2026-10-20)

Source: organisers' Kaggle notebook `azimahmadzadeh/self-evaluation-notebook`, copied verbatim to
[`official_metric.py`](official_metric.py). Input CSVs: `output/metric_check/{gt,pred}.csv`, built by
`src/make_metric_check_csvs.py` (62 multi-annotated val files; annotator set 0 = GT, set 1 = pred).

## 1. Alignment result (Phase 2, step 2A)
| Run | PQ |
|---|---|
| Official notebook on Kaggle | **0.3296** |
| Official code run locally (`official_metric.py`) | **0.3296** |
| Our `src/evaluate.py` (same files) | **0.3296** |

Notebook plots on this pair: mean IoU 0.519 and mean Dice 0.665 over all overlapping pairs (IoU > 0);
GT hits 369 / misses 71; prediction hits 365 / misses 86; 1:2 ≈ 9, 1:3 ≈ 1, 2:1 ≈ 14.
These plots count **any** overlap, unlike our IoU-based counters, so the numbers differ by design.

## 2. How the official metric works
- **GT ids are `<annotator>-<image>_<i>`** (the COCO image `id`, e.g. `010401-20110803164914Mh_1`).
  The code splits off the image id at the first `_`, then the annotator at the first `-`.
  Prediction ids are `<image>_<i>` (plain file stem).
- **Each annotator-image is scored as its own entry.** One prediction set for an image is compared
  against *every* annotator's set for that image; TP/FP/FN are pooled over all entries.
  An image labelled by 3 annotators therefore counts 3×, and its FPs are counted 3×.
- **PQ = Σ IoU(TP) / (TP + ½FP + ½FN)**, pooled over the dataset (same as SQ × RQ). Threshold is IoU **> 0.5**.
- **Matching is not one-to-one.** Every GT–prediction pair with IoU > 0.5 is a TP. A prediction is an FP
  only if it has no such pair; a GT is an FN only if it has no such pair.
- Images that have no GT rows are skipped, so predictions on them cost nothing. Predictions for images
  not in GT are ignored.
- Dice and the hit plots are diagnostics only; they do not enter the score.

## 3. Differences from our `src/evaluate.py` (measured locally)
| # | Difference | Test | Official | Ours |
|---|---|---|---|---|
| A | **Multi-annotator scoring.** Official scores against all sets; ours uses one set (`first`). | 141 val files (233 annotator-images): GT = all sets, pred = an exact copy of set 0 | **0.7394** | 1.0 |
| B | **Duplicates count twice.** Official has no one-to-one matching; ours is greedy one-to-one. | Every prediction in `pred.csv` submitted twice | **0.3729** (up from 0.3296) | 0.3296 → lower (each duplicate becomes an FP) |
| C | Images with no annotations are skipped officially; ours counts their predictions as FPs. | 0 val files have an empty set 0 | — | no effect today |

### Consequences
- **A:** Our current val PQ is over-optimistic. Even a perfect copy of one annotator scores only ~0.74
  officially, so model selection must use official-style scoring against **all** annotator sets.
  It may also help to train on all sets, or on a consensus, instead of `first` (Phase 3 ablation).
- **B:** The official metric rewards overlapping/duplicate masks. This is a loophole in the metric, not a
  modelling gain. We do **not** exploit it unless the user explicitly decides otherwise (it could
  be seen as gaming, and the organisers may fix it). Our pipeline outputs non-overlapping masks (NMS),
  so our local score still has to mirror official matching to predict the leaderboard correctly.

## 3b. Kaggle server rejects overlapping masks (2026-10-20)
First upload (`submission_baseline_r50.csv`) was refused: *"Invalid Submission! Submissions may not contain
overlapping masks. That is, predictions of the same filament may not overlap."* This is the server-side
"verification strategy" from the overview page, and it also closes loophole B above.
- Baseline CSV had 20 overlapping pairs in 18/176 images (median 792 px overlap).
- Fix: `utils.rle.remove_overlaps` (higher-score mask keeps shared pixels; masks left < `MIN_AREA` dropped),
  applied in `predict_images` and per grid cell in the sweep; `validate_submission.py` now fails on any
  overlap. `src/fix_overlaps.py` repairs an existing CSV without a GPU: 1,077 → 1,072 masks.
- Val effect not yet measured (val predictions were not saved); next Colab val run includes the fix.

## 4. Follow-ups
- [x] Official mode added: `evaluate_official()` / `evaluate_split()` in `src/evaluate.py`,
      `load_gt_entries()` in `src/annotations.py`; `tests/test_official_metric.py` reproduces 0.3296 and
      0.7394 and matches `official_metric.py` on random masks. `run_pipeline.py --step val` logs
      `official_PQ` as `val_pq`.
- [ ] Decide training-label policy (first / all sets / consensus) on official-mode val PQ (Phase 3 ablation).
- [x] User decision (2026-10-20): do **not** report the duplicate loophole; it is not exploited either.
