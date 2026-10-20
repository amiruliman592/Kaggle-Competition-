# HANDOVER — Solar Filament Segmentation 2026

Last updated 2026-10-20 (Phase 4 runs scored). Read this first in a new session, then `CLAUDE.md`, `COMPLETED_TASKS.md`
(everything finished, with numbers) and `Project_Phase_Plan.md`. Do not re-run anything recorded there.

## 1. Goal & working setup
- Kaggle instance segmentation of solar filaments in 2048×2048 GONG H-alpha images; one CSV (`filament_id,segmentation_rle`).
- Metric: Panoptic Quality (official Kaggle variant, see §3). Deadline **Nov 15, 2026** (code freeze Nov 11).
- Training runs on **Google Colab** (local GPU is a 4 GB GTX 1650; Detectron2 not installed locally — IDE
  "missing import" errors for detectron2 are expected). Claude writes code/notebooks; the user runs them and
  reports back. Colab clones branch `Dev` from GitHub (`origin` = github.com/amiruliman592/Kaggle-Competition-),
  so changes must be **pushed** before a Colab run.
- User is new to segmentation: explain in plain language. Commit/push only when the user asks.
- **Workflow rule: before every new phase (or step that runs code/Colab), first EXPLAIN the plan in plain language
  and wait for the user's explicit go-ahead. Never start executing a phase on your own.**
- Keep helper tools simple and token-cheap (user request).

## 2. Current state & achievements
- **Phases 0–3: DONE. Phase 4: both runs done and scored; Kaggle upload pending.**
- **Best model: `x101_2048`** — Mask R-CNN X101-FPN at native 2048, all annotators' labels (exp008):
  **val official PQ 0.4264, Dice 0.6417** (thr 0.6, overlap drop 0.2; PQ 0.426 flat over thr 0.6–0.8).
  Close second `r50_2048` (exp007): PQ 0.4250 (thr 0.7); thr 0.6 gives PQ 0.4231 / Dice 0.6484 (best Dice).
  Checkpoints on Drive `MyDrive/solar_filament/output/detectron2/{x101_2048,r50_2048}/model_final.pth`.
- Previous best `r50_allann` (1024): val PQ 0.3979, Dice 0.6426, **public LB 0.35** (`submission_r50_allann_pp.csv`).
- LB history: baseline 0.34 → baseline + post-processing 0.34 → r50_allann **0.35**. Val ≈ LB + 0.04–0.05, and
  val improvements carried over to LB, so model selection stays on local val.
- Phase 3 findings (`experiments/ablation_table.md`): overlap removal costs −0.004; new overlap rule
  (`OVERLAP_DROP_FRAC=0.2`) +0.003 PQ / +0.016 Dice; closing kernel, disk filter, min area, thr > 0.8: no effect;
  CLAHE + disk masking (Run A): no PQ gain; all-annotator labels (Run B): +0.008.
- Phase 4 findings: native 2048 = **+0.027 PQ** (medium and large bins both up, SQ and RQ both up); X101 adds only
  +0.0014 PQ but has the flattest threshold plateau and fewer merges. Small filaments still mostly missed (X101 2/10).
  Large bin improved, so the bigger-anchor run is not needed. A100 2048 training peaks at 20.2 GB (X101).
- Raw predictions (val + test): `Result Phase 3/Result Step 1|3/`, `Result Phase 4/` → any post-processing,
  scoring or test CSV is done locally (`src/ablate_postprocess.py`, `src/submit_from_raw.py`, `src/viewer.py`;
  the viewer only scans `Result Phase 3/`). Phase 4 scoring table: `Result Phase 4/eval_phase4.json`.
- Tests: 34 pass (`pytest tests`, ~3 min).

## 2a. Next step
- **FIRST THING NEXT SESSION (required, before anything else): ask the user for the Kaggle public LB scores of
  both Phase 4 submissions.** Do not start new work until both are recorded.
  | File | Kaggle description | Val official PQ / Dice | LB |
  |---|---|---|---|
  | `submissions/submission_x101_2048_pp.csv` | `x101_2048: Mask R-CNN X101-FPN, native 2048, ...` | 0.4264 / 0.6417 (exp008) | **pending** |
  | `submissions/submission_r50_2048_pp.csv` | `r50_2048: Mask R-CNN R50-FPN, native 2048, ...` | 0.4231 / 0.6484 (exp007, thr 0.6) | **pending** |
  Record each score in `experiments/log.csv` (`lb_score` of exp008 / exp007), `COMPLETED_TASKS.md` (Phase 4) and
  `experiments/ablation_table.md` (Phase 4 rows), and update §2 above. Expected LB ≈ 0.37–0.38 (val − 0.04–0.05);
  if the LB does not rise above 0.35, investigate the val–LB gap before Phase 5.
- Then decide (explain first, wait for "go"): (a) one run with a finer mask head (`ROI_MASK_HEAD.POOLER_RESOLUTION`
  14 → 28; Mask R-CNN draws each mask on a 28×28 grid per box, coarse for long thin filaments) to raise SQ/Dice; or
  (b) go straight to Phase 5 (Mask2Former / SAM 2 box refinement), now starting from the 2048 models.
- Phase 4 pipeline: `run_pipeline.py --backbone x101 --resolution 2048` (same flags for train/val/dump/infer);
  notebook `notebooks/colab_phase4_native.ipynb`. Compute: Colab Pro A100 80 GB, ~8 units/h; user had ~184 units
  before the Phase 4 runs (check the remaining balance in Colab).
- Later: Phase 5 (Mask2Former / SAM 2), Phase 6 (TTA/ensembling: R50+X101 at 2048 are ready members),
  Phase 7 freeze (Nov 11), Phase 8 report (Nov 15).

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
- Val–LB gap is steady (~0.04–0.05; overlap removal explained −0.004 of it).
- Small filaments (< 200 px²): still 8–10 of 10 missed on val even at 2048 (only 51 in all of train); low priority,
  worth ≤ ~0.005 PQ.
- Training-label policy settled: "all" (Run B). Consensus labels untested.
- YOLOv8-seg comparison run (user knows YOLO) — later.

## 6b. Tools
- `python src/viewer.py [--run NAME] [--split test]` — one-window viewer of saved predictions vs GT (user asked to keep it simple; don't extend unless asked).

## 7. Gotchas
- Kaggle pages and notebooks are JS-rendered: WebFetch returns only the title. Ask the user to paste content.
- When generating code/notebooks through a Bash heredoc, Python `\n` escapes break; use `chr(10)` or the Write tool,
  then compile-check every notebook cell.
- "Could not find cuda drivers" in Colab logs comes from TensorFlow, not PyTorch — harmless.
- Colab ships two OpenCV builds (same `cv2`); harmless.
- Lecture notes (`lecturenotesMachineVision/`, untracked) were reviewed; method verdicts are in
  `Project_Phase_Plan.md` §14. The W14 folder has a report template useful for Phase 8.
