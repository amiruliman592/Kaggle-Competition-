# results/ — saved predictions and scores of every run

Models are trained and run on a GPU once; their **raw predictions** (all masks with score ≥ 0.2, before any
post-processing) are saved here as `raw_preds_<run>_<val|test>.json.gz`. All fusion, post-processing and scoring is
then done on the CPU from these files, so every number in [`../experiments/ablation_table.md`](../experiments/ablation_table.md)
can be recomputed without a GPU.

| Folder | Contents | Experiments |
|---|---|---|
| `phase2/` | Baseline Mask R-CNN R50 at 1024 px: threshold sweep, validation results | exp002–003 |
| `phase3/step1/` | Baseline raw predictions, post-processing ablations | exp004 |
| `phase3/step3/` | Retraining with CLAHE + disk masking, and with all annotator sets as samples | exp005–006 |
| `phase4/` | Native 2048 models: `r50_2048` (a final model), `x101_2048` | exp007–008 |
| `phase5/step1/` | SAM 2 mask refinement (rejected) | exp009 |
| `phase5/step1b/` | First ensemble: X101 + R50 | exp010 |
| `phase5/step2/` | `x101_2048_m28` (a final model), Mask2Former (rejected), ensembles | exp011–013 |
| `phase6/Mirul/` | Noise check (bootstrap), error budgets, **flip views and fused predictions of the final submission** | exp101 |
| `phase6/Syaz/` | Training-recipe trials: second seed, longer schedule, rotations, copy-paste, self-training | exp201–205 |
| `phase7/` | Third ensemble member (rejected), models trained on all 707 files (`code_b/`, `phase7b/`), 4-model fusion (`final4/`), re-scoring (`rescore/`) | exp014–016 |
| `phase7/final_run/` | **Final submission CSV and the clean-clone reproduction run** | exp101 |

## Files of the final submission (exp101)

| File | What |
|---|---|
| `phase5/step2/raw_preds_x101_2048_m28_test.json.gz`, `phase6/Mirul/raw_preds_x101_2048_m28_flip{h,v,hv}_test.json.gz` | The 4 views of model 1 |
| `phase4/raw_preds_r50_2048_test.json.gz`, `phase6/Mirul/raw_preds_r50_2048_flip{h,v,hv}_test.json.gz` | The 4 views of model 2 |
| `phase6/Mirul/raw_preds_tta_m28_test.json.gz`, `raw_preds_tta_r50_test.json.gz` | Each model's 4 views fused |
| `phase6/Mirul/raw_preds_tta2_m28_r50_test.json.gz` | Both models fused (before post-processing) |
| `phase7/final_run/submission_exp101.csv` | The submitted CSV, built from the file above with `python src/submit_from_raw.py <file> <csv> --score-thr 0.6` |
| `phase7/final_run/submission_final.csv`, `report_test.json`, `report_val.json`, `raw_preds_final_test.json.gz` | Output of the clean-clone run of `notebooks/full_pipeline.ipynb` |
| `phase7/final_run/full_pipeline_executed_colab.ipynb` | That run's notebook exactly as downloaded from Colab |

The same files exist for the validation split (`_val` instead of `_test`).

## Older folder names

The folders were renamed on 2026-11-10. Logs and JSON files saved before that date still show the old names:

| Old name | New name |
|---|---|
| `Result Phase 2/` | `phase2/` |
| `Result Phase 3/Result Step 1/`, `.../Result Step 3/` | `phase3/step1/`, `phase3/step3/` |
| `Result Phase 4/` | `phase4/` |
| `Result Step 5/Step 1/`, `Step 1b/`, `Step 2/` | `phase5/step1/`, `step1b/`, `step2/` |
| `Result Phase 6/Mirul/`, `.../Syaz/` | `phase6/Mirul/`, `phase6/Syaz/` |
| `Result Phase 7/`, `.../Code B/`, `.../Phase7b/` | `phase7/`, `phase7/code_b/`, `phase7/phase7b/` |
