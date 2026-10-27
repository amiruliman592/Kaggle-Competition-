# HANDOVER — Solar Filament Segmentation 2026

Last updated 2026-10-27 (Phases 0–5 done by Kimee; **Phase 6 handed to Mirul and Syaz** while Kimee is on a break;
Syaz: setup + Colab sections 1–4 done, probe next; Mirul: M1 done, **M2 flip TTA = new best, LB 0.39**, M3 code
ready, Colab run next).
**This file is the team's shared memory: every member's Claude session MUST keep it current (§0.7).**
Read this first in a new session, then `CLAUDE.md`, `COMPLETED_TASKS.md` (everything finished, with numbers) and
`Project_Phase_Plan.md` (Phase 6 = the current task list). Do not re-run anything recorded there.

## 0. New teammate? Start here (Mirul, Syaz)
**First read `PHASE6_GUIDE.md`**: the plain-language version of the project and of your tasks (what, why, how long).

### 0.1 Who is who
| Person | Role | Phase 6 track |
|---|---|---|
| **Kimee** (GitHub `shamsulhakimee`) | Lead; built Phases 0–5. On a break during Phase 6; comes back **when both Phase 6 tracks are finished** for the final A100 run | — |
| **Mirul** (GitHub `amiruliman592`, repo owner) | Track M: analysis, test-time flips, external images | `Project_Phase_Plan.md` §6.2 (M1–M4) |
| **Syaz** | Track S: training recipe on T4 | `Project_Phase_Plan.md` §6.3 (S1–S5) |

Team chat = the group chat. Every push is announced there.

### 0.2 Before Phase 6 starts (Kimee, once)
- [x] Repo access: Mirul owns github.com/amiruliman592/Kaggle-Competition- and has given Kimee and Syaz write access.
- [ ] Mirul and Syaz are on the **Kaggle team** (Kaggle → competition → Team tab). Uploads count against the team's 5/day.
- [x] Kimee's Drive folder `solar_filament` (checkpoints `output/detectron2/x101_2048_m28/`, `.../r50_2048/`, needed for
      M2 TTA and M3 pseudo-labels) is shared by link. Kimee sends the link in the group chat — it is **not** in the repo (§0.6).
- When **both** tracks are done and the §2a results block is filled in, tell Kimee in the group chat: that is Kimee's
  signal to come back. Latest hand-back **Nov 6**: the A100 runs need ~2–3 days before the Nov 11 code freeze.

### 0.3 One-time setup (each member)
1. **Accounts:** GitHub (repo access already given), Kaggle (join the team), Google university account (Google AI Plus
   student plan) for Colab + Drive.
2. **Get the code + data** (data is inside the repo: ~0.7 GB, images included, so the clone takes a while):
   ```
   git clone -b Dev https://github.com/amiruliman592/Kaggle-Competition-.git Solar_Filament_Kaggle
   cd Solar_Filament_Kaggle
   ```
3. **Python on your laptop** (CPU only — scoring, ensembles, CSVs, tests). One command, from the repo folder:
   ```
   powershell -ExecutionPolicy Bypass -File setup_local.ps1     (Windows)
   bash setup_local.sh                                         (Mac/Linux)
   ```
   It finds Python 3.10–3.13 (Windows: installs 3.12 with winget if missing; then reopen the terminal and run it
   again), creates `.venv`, installs CPU PyTorch + `requirements.txt`, registers the Jupyter kernel
   "Python (solar-filament)" and runs all tests (~2–3 min, all must pass). Safe to re-run. Afterwards:
   `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Mac/Linux); in VS Code pick the `.venv` interpreter.
   Detectron2 is **not** installed locally (it only runs on Colab); "missing import detectron2" warnings in the
   editor are expected.
4. **Claude Code:** install it (VS Code extension or the `claude` CLI), open the `Solar_Filament_Kaggle` folder and
   say e.g. *"I'm Mirul, start my onboarding."* Claude reads `CLAUDE.md` automatically; it then reads this file,
   explains the project and walks you through your tasks (see `CLAUDE.md` §0).
5. **Colab:** Runtime → Change runtime type → **T4 GPU**. Notebooks mount your Drive and work in
   `MyDrive/solar_filament/` = the **team Drive folder** shortcut (§0.6); checkpoints are saved there so training
   survives disconnects. If the repo is private,
   add a GitHub token as a Colab secret named `GITHUB_TOKEN` (key icon in the left bar) — the notebooks use it to clone.
   Colab clones branch `Dev`, so **push before you run** a notebook that needs new code.

### 0.4 Git — the only commands you need
```
git pull origin Dev                 # every time BEFORE you start working
git status                          # see what you changed
git add -A
git commit -m "Mirul: M1 noise check"   # start the message with your name
git pull origin Dev                 # ALWAYS pull again right before pushing
git push origin Dev
```
Then post in the group chat: **"pushed new version: <what changed>"**. Others run `git pull origin Dev` when they see it.
- If `git pull` says **CONFLICT**: stop and ask Claude to help merge — never delete someone else's lines.
- Never use `git push --force`.
- Claude never commits or pushes on its own; it does so only when you ask.

### 0.5 Project timeline
| Phase | Dates | Who | Status |
|---|---|---|---|
| 0 Environment, tooling | Oct 18–19 | Kimee | done |
| 1 EDA, official metric, frozen val split | Oct 19–20 | Kimee | done |
| 2 Baseline Mask R-CNN R50 (LB 0.34) | Oct 20 | Kimee | done |
| 3 Pre/post-processing, all-annotator labels (LB 0.35) | Oct 20 | Kimee | done |
| 4 Native 2048 resolution, X101 (LB 0.37) | Oct 20–21 | Kimee | done |
| 5 SAM 2 ✗, ensemble ✓ (LB **0.38**), Mask2Former ✗, finer mask head | Oct 21 | Kimee | done |
| **6 Two-person sprint** (noise, TTA, training recipe, external images) | **Oct 27 – Nov 5** | **Mirul + Syaz** | **next** |
| Final 2048 runs on A100 with the Phase 6 recipe | after hand-back (target Nov 6–8) | Kimee | planned |
| 7 Freeze, 3 final submissions, latency, reproduction | Nov 9–11 (code freeze Nov 11) | all | planned |
| 8 4-page report, pipeline notebook, public repo; Google form | Nov 12–14 (deadline **Nov 15**) | all | planned |

### 0.6 Team Drive folder (Kimee's `solar_filament` folder)
Kimee shares the folder link **privately in the group chat** ("anyone with the link can edit"). **Never write the link
into any file in this repo** — the repo goes public.
- **Setup (once):** open the link while logged in to **your own** Google account → right-click the folder → Organize →
  **Add shortcut** → My Drive. Colab then sees it at `/content/drive/MyDrive/solar_filament` (the path all notebooks
  use). If you already have your own folder called `solar_filament`, rename yours first. No login to Kimee's account needed.
- **Never change or delete anything that is already there** (Kimee's checkpoints in `output/detectron2/`, the
  `Phase 3/4/5` folders). Read them only. (Kimee keeps a backup copy on Kimee's laptop, but restoring it costs time.)
- **Write only inside your own folder:** `Phase 6/Mirul/` or `Phase 6/Syaz/` (Syaz's checkpoints go to
  `Phase 6/Syaz/output/`), plus `external/` for Mirul's images + pseudo-labels. The Phase 6 notebooks are already set
  up this way. Run names still start with your name (e.g. `syaz_r50_3x`): training resumes from a folder with the
  same name, so this keeps runs apart in the logs too.

### 0.7 Keeping this file current — MANDATORY
Any member may open a fresh Claude Code session at any time (new laptop, new chat, context reset). That session only
knows what is written in the repo. So **`HANDOVER.md` must always describe the real current state.**
- **When:** after every finished task or step (M1, S2, …), after every Colab run is scored, after every Kaggle upload,
  whenever a plan or decision changes, and **before every push** that contains results. Claude does this without
  being asked; a task is not "done" until HANDOVER is updated.
- **Every session, at the start and at the end (Kimee's rule, no exceptions):** at the start, Claude checks that your
  Live status block matches `experiments/log.csv`, `COMPLETED_TASKS.md` and `git log`, and fixes it first if not.
  At the end, Claude updates your block **even if nothing finished** (half-done work, e.g. a Colab run still
  training: run name + where the checkpoints are) and asks you to commit + push it. An update that is not pushed is
  invisible to the others.
- **What:** your own **Live status** block in §2a (last done, numbers, next step, blockers, where files are); the
  §2a results block when a Phase 6 question is answered; §2 if the best model/submission changes; §6/§7 for new open
  questions or gotchas. Also the matching line in `COMPLETED_TASKS.md` and `experiments/log.csv`.
- **Rules:** edit only your own Live status block (the other member's block is theirs); keep it short and factual;
  update the "Last updated" line at the top; `git pull origin Dev` first so you don't overwrite the other's update.

## 1. Goal & working setup
- Kaggle instance segmentation of solar filaments in 2048×2048 GONG H-alpha images; one CSV (`filament_id,segmentation_rle`).
- Metric: Panoptic Quality (official Kaggle variant, see §3). Deadline **Nov 15, 2026** (code freeze Nov 11).
- Training runs on **Google Colab**. Kimee used A100s (paid units, ~95 left); Mirul and Syaz use **T4s (15 GB)**:
  R50 at 1024 trains fine (6,000 its ≈ 1.3 h); X101 at 2048 training needs 20 GB, so 2048 training is Kimee's A100 job.
  Prediction (no training) at 2048 should fit on a T4 — M2 probes this first. Claude writes code/notebooks; the
  member runs them and reports back. Colab clones branch `Dev` from GitHub (`origin` = github.com/amiruliman592/Kaggle-Competition-),
  so changes must be **pushed** before a Colab run. Mirul and Syaz use Google AI Plus student accounts (more Colab
  quota than the free tier).
- All members are new to segmentation: explain in plain language. Commit/push only when the member asks.
- **Workflow rule: before every new phase/step (or anything that runs code on Colab), first EXPLAIN the plan in plain
  language and wait for the member's explicit go-ahead. Never start a step on your own.**
- Keep helper tools simple and token-cheap (Kimee's request).

## 2. Current state & achievements
- **Phases 0–5: DONE** (Phase 5: SAM 2 and Mask2Former rejected, m28 head kept).
- **Best submission (Phase 6 M2, 2026-10-27): flip TTA of x101_2048_m28 + r50_2048** (exp101): each model on the
  original + h/v/hv-flipped image, 4 views fused (`src/ensemble.py`, IoU 0.5 union), then the 2 models fused (IoU 0.3
  inter), close 5, min area 50, thr 0.6, overlap drop 0.2. **Val official PQ 0.4480, Dice 0.6520; public LB 0.39**.
  Fused raw files in `Result Phase 6/Mirul/raw_preds_tta2_m28_r50_{val,test}.json.gz`; CSV:
  `python src/submit_from_raw.py "Result Phase 6/Mirul/raw_preds_tta2_m28_r50_test.json.gz" submissions/submission_tta2_m28_r50_pp.csv --score-thr 0.6`.
- Previous best: **ensemble x101_2048 + r50_2048** (exp010, `src/ensemble.py`): masks of both models with IoU ≥ 0.5
  are one filament, mask = union, score = mean over both models (a missing model adds 0); then close 5, min area 50,
  thr 0.6, overlap drop 0.2. **Val official PQ 0.4366, Dice 0.6505; public LB 0.38**
  (`submissions/submission_ens_x101_r50_2048_pp.csv`; fused raw files in `Result Step 5/Step 1b/`).
- Best single model on val `x101_2048_m28` (exp012, 56×56 mask head): val PQ 0.4385, Dice 0.6408, LB 0.37; ensemble
  m28 + r50 (exp013) val 0.4404, LB 0.37. Previous best single `x101_2048` (exp008): val PQ 0.4264, Dice 0.6417, LB 0.37. `r50_2048` (exp007): val PQ 0.4231 /
  Dice 0.6484 at thr 0.6, LB 0.36. Checkpoints in the team Drive folder `solar_filament/output/detectron2/<name>/model_final.pth` (§0.6; read only).
- 1024 reference for Phase 6 recipe trials: `r50_allann` (exp006) val PQ 0.3979, Dice 0.6426, LB 0.35
  (raw preds `Result Phase 3/Result Step 3/`).
- LB history: 0.34 (baseline) → 0.34 (post-proc) → 0.35 (r50_allann) → 0.36 (r50_2048) → 0.37 (x101_2048) →
  **0.38 (ensemble)**. Val ≈ LB + 0.05–0.06 on every upload and val gains carried over until Phase 5, so model
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
  - Phase 5 step 2 (exp011–013): finer mask head +0.012 val but LB 0.37; Mask2Former 0.4020 (twice the merges,
    lowers every ensemble) → dropped.
- Raw predictions (val + test) for all models are **in git**: `Result Phase 3/`, `Result Phase 4/`, `Result Step 5/`.
  All scoring, post-processing and CSVs are done locally on CPU: `src/ablate_postprocess.py`, `src/ensemble.py`,
  `src/sam_refine.py`, `src/submit_from_raw.py`, `src/viewer.py` (viewer only scans `Result Phase 3/`).
- Tests: 53 pass (`pytest tests`, ~1.5–3 min; checked 2026-10-26 in a fresh `.venv` made by `setup_local.ps1`).
- Kimee's Colab balance: ~95 units (estimate, to be confirmed), reserved for the final A100 runs (Nov 6–8).

## 2a. Next step (start here): Phase 6, Oct 27 – Nov 5
**Why:** since Phase 4 every gain (+0.002–0.01) is below the noise of the 141-image val and the ~90-image, 2-dp LB:
exp012/exp013 were better on val but scored LB 0.37 < exp010's 0.38. We were tuning on noise. Target: LB 0.40–0.42
(val ~0.46–0.48), stretch ~0.45. Idea: change what was never touched since Phase 2 — the training recipe and the
amount of data — and measure the noise so only real gains are kept.

**Full task list with inputs, outputs and "done when": `Project_Phase_Plan.md` → Phase 6. Plain-language version:
`PHASE6_GUIDE.md`.** Ready to run (built by Kimee's session, 2026-10-26): `notebooks/colab_phase6_syaz.ipynb` (S1 + S2),
`notebooks/colab_phase6_mirul.ipynb` (M2), `run_pipeline.py --seed` and `--flip h|v|hv` (dump only),
`setup_local.ps1` / `setup_local.sh`. Short version:

| Mirul (track M) | Syaz (track S) — R50 at 1024 on T4, one change at a time vs exp006 |
|---|---|
| **M1** noise check (paired bootstrap, simulated LB) + error budget — CPU, first, Oct 27–29 | **S1** seed repeat of exp006 → training noise (~1.3 h) |
| **M2** flip TTA of Kimee's m28 + r50 2048 checkpoints (T4 inference) → fuse with `src/ensemble.py`; upload if above threshold | **S2** 3× schedule, `--max-iter 18000` (~4 h) |
| **M3** ~3,000 external GONG H-alpha images (no test/val days) + pseudo-labels from m28 → team Drive `external/` | **S3** + 90° rotations + scale jitter |
| **M4** per-station / per-size table (if time) | **S4** + copy-paste of GT filaments · **S5** + Mirul's pseudo-labels |

**Shared rules:** pull before work and before push, announce pushes in the group chat; results in
`Result Phase 6/Mirul/` and `Result Phase 6/Syaz/`; explicit experiment IDs (**Mirul exp101+, Syaz exp201+**,
Kimee exp014+; pass `exp_id=` to `log_experiment`); Kaggle uploads only from Mirul (≤ 2/day) for candidates above
the M1 threshold; each member logs in their own Phase 6 subsection of `COMPLETED_TASKS.md`.

**Live status — each member updates ONLY their own block, after every step (mandatory, §0.7):**

> **Mirul** — last updated: 2026-10-27 (M1 + M2 done; M3 code ready, Colab run not started)
> - Done: local setup (clone, `.venv`, tests pass). **M1 bootstrap** (`src/bootstrap.py`): a val gain must be
>   **≥ ~0.010 PQ** to be real (LB ≥ ~0.015, i.e. +0.02 at 2 dp); exp012/exp013 gains over exp010 are noise.
>   **M1 error budget** (`src/error_budget.py`) of exp010: biggest losses = 246 true filaments scored < 0.6
>   (confidence, ≤ +0.063 if fixed) and 341 real false alarms (≤ +0.051); 181 never found (≤ +0.047); 146 near
>   misses; 186 FPs drawn by another annotator (label noise). Tables: `experiments/ablation_table.md` (Phase 6 M1a/M1b)
>   **M2 flip TTA** (exp101, **new best: val 0.4480, LB 0.39**): Kimee's m28 + r50 on h/v/hv-flipped val + test
>   (T4); two-stage fusion = per model 4 views (IoU 0.5 union), then m28 + r50 (IoU 0.3 inter), thr 0.6.
>   +0.0116 val vs exp010 (above the bar). One 8-view fusion gives no gain. Table: ablation_table Phase 6 M2
> - **M3 code ready** (not run): source = GONG archive product `hag` (same frames/names as the competition JPEGs,
>   corr ≥ 0.99, archive JPEG quality ~75 vs ~90). Excludes every val 27-day block + test day ±1 (3,310 allowed
>   days 2011–2022), 1 frame/day, stations balanced, QC vs train range. Labeller **m28 alone, thr 0.85** (val
>   precision 0.84, recall 0.53, 4.7 masks/img; 8-view TTA only ~2 pts better for 8× the T4 time; table:
>   ablation_table Phase 6 M3). `src/external/download_gong.py`, `src/external/make_pseudo_labels.py`,
>   `run_pipeline.py --step dump --image-dir`, `tests/test_external.py`, notebook `notebooks/colab_phase6_mirul_m3.ipynb`.
> - Now / next: push → run `colab_phase6_mirul_m3.ipynb` on T4 (~3–4 h) → check Step D val PQ on archive copies
>   (≈ 0.4355 expected) → hand pseudo-labels to Syaz (S5) → M4
> - Blockers / waiting on: Colab writes to the team Drive fail for me (§7) — the M3 notebook saves to my own
>   `MyDrive/m3_mirul/` and then tries the team `external/` (checked)
> - Files: `Result Phase 6/Mirul/` (bootstrap, error budget, 12 flip raw files, fused `raw_preds_tta*`, grids,
>   logs); CSV `submissions/submission_tta2_m28_r50_pp.csv` (gitignored, on Mirul's laptop; rebuild with §2).
>   Laptop: Mirul works only in `Downloads/Solar_Filament_Kaggle` (own `.venv` + M2 CSV; the old
>   `Downloads/GENE/...` clone is retired)

> **Syaz** — last updated: 2026-10-27 (S1 trained; S2 interrupted)
> - Done: local setup (§0.3 steps 1–4): clone, `setup_local.ps1`, all 53 tests pass; team Drive shortcut added
>   (university account, `output/detectron2` visible); `colab_phase6_syaz.ipynb` opened from GitHub (Dev) on a T4,
>   copy saved in own Drive; sections 1–4 run without errors (repo is public: no `GITHUB_TOKEN` needed);
>   **Step 0 probe passed** (T4, 20 its in 1.1 min, peak GPU memory 2.7 GB of 15; writes to Drive `Phase 6/Syaz/output/`)
> - Done (2026-10-27): **S1 trained + dumped** on Colab T4 (`syaz_r50_seed2`, `--seed 7`): val 1,771 raw masks
>   (0.56 s/img), test 2,368 (0.57 s/img), dump peak GPU 2.2 GB. Both raw files downloaded to `Result Phase 6/Syaz/`
>   and validated (141 val / 180 test images = same sets as exp006, all masks 2048², decodable, scores 0.20–1.00);
>   **not yet scored**.
> - Now / next: (1) score S1 vs exp006 on the laptop, log exp201. (2) **S2 (`syaz_r50_3x`) started, then stopped:
>   Colab GPU usage limit** ("Cannot connect to GPU backend"). Syaz continues on another Google account (needs the
>   team-folder shortcut there). Newest checkpoint in Drive `Phase 6/Syaz/output/detectron2/syaz_r50_3x/` =
>   **`model_0011999.pth`** (334.8 MB; 12,000 of 18,000 its done). Resume: sections 1–4 → S2 cell (unchanged,
>   `--max-iter 18000`); Detectron2 reads `last_checkpoint` and restores model + optimizer + LR scheduler + iteration,
>   so it trains its 12,000 → 17,999 only (~1.3 h), then dumps val/test.
> - Blockers / waiting on: Kaggle team membership **pending** (invite/merge requested from a teammate, 2026-10-27;
>   no uploads in track S)
> - Files: `Result Phase 6/Syaz/raw_preds_syaz_r50_seed2_{val,test}.json.gz`; Drive `Phase 6/Syaz/`

**Hand-back to Kimee (when both tracks are done; target Nov 5, latest Nov 6):** fill in the block below, so Kimee can
start the A100 run without asking, then tell Kimee in the group chat.

> **Phase 6 results (to be filled by Mirul and Syaz):**
> - Noise threshold — val: ~0.010 PQ (paired bootstrap, M1) / LB: ~0.015 (≈ +0.02 at 2 dp) ; training noise (S1): …
> - Error budget (main loss): confidence — 246 true filaments scored < 0.6 (≤ +0.063), then 341 real FPs (≤ +0.051), 181 never found (≤ +0.047); ablation_table Phase 6 M1b
> - TTA: **adopted** — two-stage flip TTA m28 + r50, val PQ 0.4480 (+0.0116 vs exp010), **LB 0.39**, CSV `submissions/submission_tta2_m28_r50_pp.csv` (exp101). For the final run: predict each new 2048 model on h/v/hv flips too
> - Recipe for the final 2048 run (exact flags): …
> - Pseudo-labels: team Drive `external/…`, size …, verdict from S5: …
> - Open problems: …

## 3. Competition rules & metric (sources: `Solar Filament Overview.txt`, `SelfEvaluationResult/`)
- **Final package (Nov 15, Google form):** 4-page report (Overleaf template, PDF); **public** Git repo with pinned
  `requirements.txt` and a notebook showing the whole pipeline; no extra files needed to reproduce. Repo stays
  public until winners are announced (Nov 30).
- **Judging:** 70% quantitative — PQ, mean Dice (`torchmetrics.segmentation.DiceScore`), Dice/IoU distributions,
  1:many / many:1 counts. 30% qualitative — pipeline description, visual mask quality, code quality.
  LB rank is only "one of the considerations".
- **Submissions: max 5 per day per team; at the end 3 are selected as final.** Public LB = ~50% of test images, shown to 2 decimals.
- **Kaggle rejects any file with overlapping masks in one image.** `predict_images` makes masks disjoint
  (higher score wins) and `validate_submission.py` fails on overlaps; repair old CSVs with `src/fix_overlaps.py`.
- Inference may use only the test H-alpha images; no extra ground-truth metadata for training.
- **External unlabelled images are allowed** (Kaggle forum, competition host Azim Ahmadzadeh, ~2026-10-21, answering
  a question about unlabelled KSO/Meudon/BBSO H-alpha images for self-supervised pretraining): *"Your method is OK.
  Please make sure that the new set of observations does not include the test images (or observations within
  +/- 1 hour of those timestamps)."* Our rule for M3 is stricter: skip every **day** with a test or val image, and
  never use external labels (MAGFiLO masks, filament catalogues). Pseudo-labels come from our own model.
- **Official PQ** (`evaluate_official`; code + analysis in `SelfEvaluationResult/FINDINGS.md`): every annotator set
  is its own GT entry, scored against the image's predictions and pooled; IoU > 0.5; non-unique matching.
  Our implementation matches Kaggle exactly (0.3296 = 0.3296). **Model selection uses `official_PQ`.**

## 4. Leaderboard context (2026-10-21)
- #1 IONode Online 0.69 with only 15 entries, then 0.64 (123), 0.62 ×3, 0.60–0.61.
- Treat the top as **not comparable**: human inter-annotator PQ is only ~0.336, a perfect copy of one annotator scores
  only 0.74 on val, and the organisers warned (Aug 20) about participants "hacking the system" ("Any PQ > 0.35 is of
  great value"). Likely cause (unverified): the public MAGFiLO dataset (1,593 labelled GONG images) contains the test
  ground truth.
- **Do not use** either of these: (a) external MAGFiLO labels; (b) copying train labels onto the 31 test images taken
  on the same day as a train image. Both break the rules.

## 5. Key data facts (not obvious from the code)
- JSON has 1,154 entries for **707 files**: 296 files annotated 2–3× independently (COCO image id `<annotator>-<stem>`).
  Training uses every annotator set as its own sample (`--label-policy all`, 921 samples); evaluation uses all sets.
- Human ceiling: inter-annotator PQ ≈ 0.336 (SQ 0.63, RQ 0.53, Dice 0.61).
- **Frozen split** `splits/val_ids.json` (566/141, 27-day blocks against same-day leakage). Never regenerate
  (sklearn versions produce different folds).
- Test: 180 files, none in train. File names are `YYYYMMDDhhmmss<station>h`; station = filename char 15
  (B, C, L, M, T, U), roughly balanced.
- Smallest selected GT 60 px²; only 51 instances < 200 px². Single class (`NUM_CLASSES=1`).
- Training recipe so far (`src/train.py`, `src/config.py`): 6,000 its, batch 2, LR 0.0025, LR steps at 60%/85% of
  `max_iter` (they scale with `--max-iter`), augmentations = h/v flips + brightness/contrast only, seed 42.

## 6. Open questions / TODO
- Dice settings of the organisers are unpublished; we log `official_Dice_samplewise` (torchmetrics default) and
  report the other variants too (FINDINGS §3c). Revisit if the organisers publish their Dice code.
- Val–LB gap is steady (~0.05–0.06 over 7 uploads). Likely causes: public LB is only ~90 images (noisy, 2 dp); test
  may be harder or have more multi-annotated images; slight optimism from tuning thr on val. Not a scorer bug
  (scorer = Kaggle code, 0.3296 check). M1 quantifies the LB noise.
- Small filaments (< 200 px²): still 8–10 of 10 missed on val even at 2048; low priority, worth ≤ ~0.005 PQ.
- YOLOv8-seg comparison run — dropped unless Phase 6 leaves time.

## 6b. Tools
- `python src/viewer.py [--run NAME] [--split test]` — one-window viewer of saved predictions vs GT (keep it simple;
  don't extend unless asked).

## 7. Gotchas
- Kaggle pages and notebooks are JS-rendered: WebFetch returns only the title. Ask the member to paste content or a screenshot.
- When generating code/notebooks through a Bash heredoc, Python `\n` escapes break; use `chr(10)` or the Write tool,
  then compile-check every notebook cell.
- "Could not find cuda drivers" in Colab logs comes from TensorFlow, not PyTorch — harmless.
- Colab ships two OpenCV builds (same `cv2`); harmless.
- Lecture notes (`lecturenotesMachineVision/`, in `.gitignore` — the repo must go public) were reviewed; method
  verdicts are in `Project_Phase_Plan.md` §14. Only on Kimee's laptop.
- Kimee's laptop only: Avast intercepts Python HTTPS (`CERTIFICATE_VERIFY_FAILED`); download with `curl` instead.
  Kimee also has a local alias `git cm "msg"` (not in the repo).
- Never list/read the image folders or print the 48 MB annotation JSON (CLAUDE.md token rules).
- Competition image files end in **`.jpeg`** (not `.jpg`); GONG archive files (M3) end in `.jpg`. The archive
  directory pages contain a `/secret0x2112` link (crawler trap) — never follow it; `download_gong.py` only reads
  `<stem>.jpg` links. The archive sometimes times out from Colab (M3 step A, 2026-10-27): `download_gong.py` now
  skips a timed-out day/file, retries it once, and a rerun of the cell resumes where it stopped.
- **Writes to the team Drive can silently fail (Mirul, M2, 2026-10-27).** `colab_phase6_mirul.ipynb` printed
  "saved to …/Phase 6/Mirul" for all 6 flip runs, yet that folder stayed empty (in Colab `os.listdir` and on
  drive.google.com, also hours later); the files existed only in `/content/out`. Likely cause: writing through the
  shortcut into Kimee's shared folder. Workaround: before disconnecting, check `os.listdir(MY_DIR)` and, if empty,
  zip + `files.download` the files from `/content/out`. **Syaz: check that your checkpoints really appear in
  `Phase 6/Syaz/output/` early in a training run** — otherwise a disconnect loses the run.
