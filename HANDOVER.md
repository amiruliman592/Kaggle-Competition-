# HANDOVER — Solar Filament Segmentation 2026

Last updated 2026-10-20 (end of Phase 2). Read this first in a new session, then `CLAUDE.md`, `COMPLETED_TASKS.md`
(everything finished, with numbers) and `Project_Phase_Plan.md`. Do not re-run anything recorded there.

## 1. Goal & working setup
- Kaggle instance segmentation of solar filaments in 2048×2048 GONG H-alpha images; one CSV (`filament_id,segmentation_rle`).
- Metric: Panoptic Quality (official Kaggle variant, see §3). Deadline **Nov 15, 2026** (code freeze Nov 11).
- Training runs on **Google Colab** (local GPU is a 4 GB GTX 1650; Detectron2 not installed locally — IDE
  "missing import" errors for detectron2 are expected). Claude writes code/notebooks; the user runs them and
  reports back. Colab clones branch `Dev` from GitHub (`origin` = github.com/amiruliman592/Kaggle-Competition-),
  so changes must be **pushed** before a Colab run.
- User is new to segmentation: explain in plain language. Commit/push only when the user asks.

## 2. Current state
- **Phases 0, 1, 2: DONE.** Phase 2's optional classical baseline was skipped (can be added later as a report row).
- Baseline Mask R-CNN R50-FPN (exp003): **val official PQ 0.3905** at score threshold 0.8; **public LB 0.34**.
  Raw Colab outputs in `Result Phase 2/`; checkpoint on the user's Google Drive
  (`MyDrive/solar_filament/output/detectron2/baseline_r50/model_final.pth`).
- Tests: 23 pass locally (`pytest tests`).
- **Phase 3 in progress:** steps 0–2 done. Post-processing best: val official PQ 0.3899 / Dice 0.6349 (thr 0.65, overlap drop 0.2; `experiments/ablation_table.md`). Raw baseline predictions in `Result Phase 3/` — any post-processing or test CSV can be redone locally (`src/ablate_postprocess.py`, `src/submit_from_raw.py`). Step 3 (approved) code + `notebooks/colab_phase3_retrain.ipynb` ready: Run A `r50_clahe_disk`, Run B `r50_allann`; when raw preds arrive, score with `src/ablate_postprocess.py`-style evaluation (same post-processing as B0': thr 0.65, drop 0.2) and add rows to `experiments/ablation_table.md`.

## 3. Competition rules & metric (sources: `Solar Filament Overview.txt`, `SelfEvaluationResult/`)
- **Final package (Nov 15, Google form):** 4-page report (Overleaf template, PDF); **public** Git repo with pinned
  `requirements.txt` and a notebook showing the whole pipeline; no extra files needed to reproduce. Repo stays
  public until winners are announced (Nov 30).
- **Judging:** 70% quantitative — PQ, mean Dice (`torchmetrics.segmentation.DiceScore`), Dice/IoU distributions,
  1:many / many:1 counts. 30% qualitative — pipeline description, visual mask quality, code quality.
  LB rank is only "one of the considerations".
- **Submissions: max 5 per day; at the end 3 are selected as final.** Public LB = ~50% of test images, shown to 2 decimals.
- **Kaggle rejects any file with overlapping masks in one image.** `predict_images` makes masks disjoint
  (higher score wins) and `validate_submission.py` fails on overlaps; repair old CSVs with `src/fix_overlaps.py`.
- Inference may use only the test H-alpha images; no extra ground-truth metadata for training.
- **Official PQ** (`evaluate_official`; code + analysis in `SelfEvaluationResult/FINDINGS.md`): every annotator set
  is its own GT entry, scored against the image's predictions and pooled; IoU > 0.5; non-unique matching.
  Our implementation matches Kaggle exactly (0.3296 = 0.3296). **Model selection uses `official_PQ`.**

## 4. Leaderboard context (2026-10-20)
- Top teams show 0.60–0.64 with 89–123 entries; we are ~578th at 0.34 in a large cluster of 0.34 teams.
- Treat the top as **not comparable**: human inter-annotator PQ is only ~0.336, and the organisers warned (Aug 20)
  about participants "hacking the system" ("Any PQ > 0.35 is of great value"). Likely cause (unverified): the
  public MAGFiLO dataset contains the test ground truth.
- **Do not use** either of these: (a) external MAGFiLO labels; (b) copying train labels onto the 31 test images taken
  on the same day as a train image. Both break the rules.
- Realistic targets: val official PQ 0.42–0.45, public LB > 0.35, Dice kept high.

## 5. Key data facts (not obvious from the code)
- JSON has 1,154 entries for **707 files**: 296 files annotated 2–3× independently (COCO image id `<annotator>-<stem>`).
  Training uses ONE set per file (`ANNOTATOR_POLICY="first"`, 5,251 instances); evaluation uses all sets.
- Human ceiling: inter-annotator PQ ≈ 0.336 (SQ 0.63, RQ 0.53, Dice 0.61).
- **Frozen split** `splits/val_ids.json` (566/141, 27-day blocks against same-day leakage). Never regenerate
  (sklearn versions produce different folds).
- Test: 180 files, none in train. Stations = filename char 15 (B, C, L, M, T, U), roughly balanced.
- Smallest selected GT 60 px²; only 51 instances < 200 px². Single class (`NUM_CLASSES=1`).

## 6. Open questions / TODO
- Dice settings of the organisers are unpublished; we log `official_Dice_samplewise` (torchmetrics default) and
  report the other variants too (FINDINGS §3c). Revisit if the organisers publish their Dice code.
- **Val–LB gap (0.39 vs 0.34):** possible causes — public LB noise (~90 images), overlap removal not yet measured on
  val, test distribution, 2-dp rounding. The next Colab val run includes overlap removal; log val→LB offset per upload.
- Small filaments (< 200 px²): all 10 missed on val → Phase 4 tiling.
- Training-label policy (first / all sets / consensus) — Phase 3 ablation.
- YOLOv8-seg comparison run (user knows YOLO) — later.

## 7. Gotchas
- Kaggle pages and notebooks are JS-rendered: WebFetch returns only the title. Ask the user to paste content.
- When generating code/notebooks through a Bash heredoc, Python `\n` escapes break; use `chr(10)` or the Write tool,
  then compile-check every notebook cell.
- "Could not find cuda drivers" in Colab logs comes from TensorFlow, not PyTorch — harmless.
- Colab ships two OpenCV builds (same `cv2`); harmless.
- Lecture notes (`lecturenotesMachineVision/`, untracked) were reviewed; method verdicts are in
  `Project_Phase_Plan.md` §14. The W14 folder has a report template useful for Phase 8.
