# HANDOVER — Solar Filament Segmentation 2026

Last updated 2026-10-21 (Phase 5 steps 1/1b done, step 2 ready to run). Read this first in a new session, then `CLAUDE.md`, `COMPLETED_TASKS.md`
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
- **Phases 0–4: DONE. Phase 5 in progress** (steps 1 and 1b done, step 2 code ready but not run).
- **Best submission: ensemble x101_2048 + r50_2048** (exp010, `src/ensemble.py`): masks of both models with IoU ≥ 0.5
  are one filament, mask = union, score = mean over both models (a missing model adds 0); then close 5, min area 50,
  thr 0.6, overlap drop 0.2. **Val official PQ 0.4366, Dice 0.6505; public LB 0.38**
  (`submissions/submission_ens_x101_r50_2048_pp.csv`; fused raw files in `Result Step 5/Step 1b/`).
- Best single model `x101_2048` (exp008): val PQ 0.4264, Dice 0.6417, LB 0.37. `r50_2048` (exp007): val PQ 0.4231 /
  Dice 0.6484 at thr 0.6, LB 0.36. Checkpoints on Drive `MyDrive/solar_filament/output/detectron2/<name>/model_final.pth`.
- LB history: 0.34 (baseline) → 0.34 (post-proc) → 0.35 (r50_allann) → 0.36 (r50_2048) → 0.37 (x101_2048) →
  **0.38 (ensemble)**. Val ≈ LB + 0.05–0.06 on every upload and val gains carry over (same ranking), so model
  selection stays on local val.
- Findings so far (details and numbers: `experiments/ablation_table.md`):
  - Phase 3: overlap rule `OVERLAP_DROP_FRAC=0.2` +0.003 PQ / +0.016 Dice; all-annotator labels +0.008; closing
    kernel, disk filter, min area, thr > 0.8, CLAHE + disk masking: no gain.
  - Phase 4: native 2048 +0.027 PQ (medium + large bins, SQ + RQ); X101 +0.0014 over R50 at 2048. Small
    filaments still mostly missed (low value, ≤ ~0.005 PQ).
  - Phase 5 step 1 (exp009): **SAM 2 box refinement rejected** — every variant loses PQ and Dice (best 0.4076).
    SAM draws ~15% tighter than the annotators: outline quality is limited by the labels, not mask sharpness.
  - Phase 5 step 1b (exp010): **ensembling works** (+0.010 PQ, +0.009 Dice, LB +0.01). Dilating masks 1–2 px
    hurts, so the gain is from model agreement, not mask size. Every new model should be tried as an extra member.
- Raw predictions (val + test) for all models: `Result Phase 3/`, `Result Phase 4/`, `Result Step 5/`. All scoring,
  post-processing and CSVs are done locally on CPU: `src/ablate_postprocess.py`, `src/ensemble.py`,
  `src/sam_refine.py`, `src/submit_from_raw.py`, `src/viewer.py` (viewer only scans `Result Phase 3/`).
- Tests: 43 pass (`pytest tests`, ~1.5 min).
- Colab balance 2026-10-21: **156.94 units**, minus ~1 for the SAM 2 run (T4 ≈ 1 unit/h, A100 ≈ 8 units/h).

## 2a. Next step (start here)
1. **Ask the user whether the Phase 5 step 2 A100 run is done.** If not yet run, nothing else is blocked on it;
   the user said they will run it later. Instructions for the user:
   - `git push origin Dev` must be done first (check `git status` / `git log origin/Dev`: the step 2 code and
     the LB 0.38 records must be on GitHub, otherwise Colab runs old code).
   - Open `notebooks/colab_phase5_a100.ipynb`, runtime **A100 + High-RAM**, Run all (~7 h, ~55–60 units).
     Step 0 probes both models (~15 min); step 1 Mask2Former `m2f_swin_s` (length auto-capped at ~4 h from the
     measured s/it; if its probe fails, step 1 is skipped); step 2 `x101_2048_m28` (X101 at 2048 with the finer
     56×56 mask head, `--mask-res 28`, ~3 h). Raw files go to `MyDrive/solar_filament/Phase 5/` after each step;
     the user puts them in `Result Step 5/Step 2/`. Disconnect the runtime afterwards.
2. **When the files arrive, score locally** (CPU, 0 units):
   - each new model alone, like Phase 4 (close 5, min area 50, drop 0.2, thr 0.3–0.9 — Mask2Former scores have a
     different scale, so sweep wider) with a short script: `postprocess.close_all` → `postprocess.select` →
     `ablate_postprocess.score` (official PQ + Dice), plus `evaluate_split` for size bins and frag/merge counts;
   - as extra ensemble members: `python src/ensemble.py ablate --raw <x101 val> <r50 val> <new val> --out ...`
     (with 3+ models, also check whether `match_iou`/thresholds need re-tuning: score = mean over all models);
   - record exp011+ in `experiments/log.csv`, `experiments/ablation_table.md`, `COMPLETED_TASKS.md`, then build the
     best CSV with `src/ensemble.py export` + `src/submit_from_raw.py` and ask the user to upload it.
3. **Phase 5 gate (Nov 4):** Mask2Former beats 0.4264 alone → primary model; within 0.015 or helps the ensemble →
   keep as member; otherwise drop. U-Net fallback only if Mask2Former fails to train by ~Nov 2.
4. Then **Phase 6** (explain first, wait for "go"): TTA (horizontal/vertical flips, needs a GPU run per model) and
   the final ensemble; anti-fragmentation merging only if val shows many 1→N cases. Phase 7 freeze **Nov 11**,
   Phase 8 report + public repo **Nov 14/15**.

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
- Val–LB gap is steady (~0.05–0.06 over 5 uploads; overlap removal explained −0.004 of it). Likely causes: public LB
  is only ~90 images (noisy, 2 dp); test may be harder or have more multi-annotated images (official PQ pools every
  annotator set, so disagreement lowers the score); slight optimism from tuning thr on val. Not a scorer bug
  (scorer = Kaggle code, 0.3296 check).
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
- Lecture notes (`lecturenotesMachineVision/`, now in `.gitignore` — the repo must go public) were reviewed; method
  verdicts are in `Project_Phase_Plan.md` §14. The W14 folder has a report template useful for Phase 8.
- Locally, Python HTTPS downloads (Hugging Face) fail with `CERTIFICATE_VERIFY_FAILED` because Avast intercepts
  HTTPS; `curl` works. Download weights with curl into the scratchpad and pass the folder (e.g.
  `python src/m2f.py ... --pretrained <dir>`). Colab is not affected.
- Repo alias (user's request, in `.git/config`, not committed): `git cm "msg"` = `git add -A && git commit -m`.
  The user commits and pushes themselves; give them the command instead of running it.
