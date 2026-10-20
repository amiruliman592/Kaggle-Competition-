# HANDOVER — Solar Filament Segmentation 2026

Written 2026-10-19 at the end of session 1 (Phase 0 + 1 "foundation"). Read this first in a new session, then `CLAUDE.md` and `Project_Phase_Plan.md`.

## 1. Goal
- Kaggle task: **instance segmentation** of solar filaments in 2048×2048 GONG H-alpha images. One mask per filament, submitted as RLE in one CSV (`filament_id,segmentation_rle`).
- Metric: Panoptic Quality (PQ = SQ × RQ, match at IoU > 0.5). Fragmenting one filament or merging two is penalised. Target PQ > 0.35.
- Deadline Nov 15, 2026 (code freeze Nov 11). Final deliverables also need a public repo, notebook, requirements.txt and a short report (30% of grade is qualitative).
- Training must run on **Google Colab** (user's local GPU is a 4 GB GTX 1650; Detectron2 is not installed locally). Claude gives the user code/instructions; the user runs them and reports back.

## 2. Current state (updated 2026-10-19, after Colab dry run)
- **Phase 0: DONE.** Colab T4 dry run passed end to end (exp001 in `experiments/log.csv`: 5 iters, PQ 0, Dice 0.0036, 1.67 s/img — values meaningless, plumbing proven). Detectron2 built and imported on Colab. `requirements-lock.txt` saved from that run (torch 2.11+cu130, Python 3.13, Detectron2 commit `1e3e13b`).
- **Phase 1: done.** The one open gate (local PQ vs the organisers' Kaggle `self-evaluation-notebook`, ±0.001) was **moved into Phase 2 by the user** — it is Phase 2 step 2A below. Still unverified.
- **Phase 2: baseline trained and validated** (2026-10-20, Colab T4). Val official PQ **0.3905** at score 0.8 (exp003; 0.3776 at the default 0.5). Results in `Result Phase 2/` and `COMPLETED_TASKS.md`. Remaining Phase 2 gate: Kaggle upload + LB score (recommended now; user to check remaining submissions). Optional: classical baseline row.
- **Open TODO — Dice alignment:** our `mean_semantic_Dice` (0.609 on exp003) is NOT yet verified against `torchmetrics.segmentation.DiceScore` as the organisers use it (unknown settings: per-image vs pooled, binary union masks?). Mirror it like the PQ check: compute both on `output/metric_check/*.csv` and on val predictions; add a test. Do this early in Phase 3 since Dice is graded.
- Tests: 14/14 locally. On Colab 13/14 passed; the failure was `test_frozen_split_is_reproducible` (sklearn 1.6.1 on Colab vs 1.8.0 locally produces different folds). Replaced by `test_regenerated_split_is_valid`. The frozen file is the source of truth — never regenerate it.
- EDA on Colab matches local numbers.
- Commits are local only on `Dev` (user asked to commit locally first; push not yet requested). The user ran the notebook from a clone, so the remote apparently had the code — check `git status`/`git log origin/Dev` before assuming. Remote `origin` = github.com/amiruliman592/Kaggle-Competition-.
- No real model trained, no Kaggle submission made yet.

## 2b. Competition rules (from `Solar Filament Overview.txt` = saved Kaggle overview page; Kaggle pages can't be WebFetched)
- **Nov 15, 2026:** final package via the organisers' Google form: 4-page report (Overleaf template, PDF), **public** Git repo with `requirements.txt` (pinned versions) and a Jupyter notebook showing the whole pipeline; no extra files needed to reproduce predictions. Repo must be public at competition close until winners are announced (Nov 30).
- **Judging:** 70% quantitative — PQ, **mean Dice (torchmetrics `DiceScore`)**, Dice and IoU distributions, 1:many / many:1 counts (= the self-evaluation notebook plots). 30% qualitative — pipeline description, visual mask quality on H-alpha images, code quality (modularity, docs). LB rank is "one of the considerations".
- **Leaderboard:** shows a score on ~50% of the test images; the rest is private. The page says both "mean Dice" (scoreboard) and PQ (ranking) — check which one the first upload shows.
- Server has "further verification strategies" for edge cases; Aug 20 announcement warns against "hacking the system". "Any PQ > 0.35 is of great value."
- Inference may use only the provided test H-alpha images; no extra ground-truth metadata for training.
- Daily submission limit / number of final selections: not on the page — read the "submissions remaining" note on the Kaggle Submit page.

## 3. Key findings (not obvious from the code)
- **Multi-annotator structure:** the JSON has 1,154 image entries for only **707 files**. 296 files were annotated 2–3 times independently (each set is a separate COCO "image", ids like `050103-20111116063134Lh`). Overlapping sets agree with median IoU ~0.56. Everything is keyed by file stem; `src/annotations.py` picks ONE set per file (`ANNOTATOR_POLICY="first"` → 5,251 instances, ~7.4/file). The 8,199 / ~7.1 figures in the plan counted all sets.
- **Human ceiling:** inter-annotator PQ ≈ **0.336** (SQ 0.63, RQ 0.53, Dice 0.61) over 447 comparisons. So PQ 0.35 is about human-agreement level; RQ (count/fragmentation) matters as much as outline precision.
- **Leakage:** 276 of 656 training dates have another image within 1 day. Split therefore groups by 27-day blocks.
- Test set: 180 files, no overlap with train files; 31 test dates also occur in train dates.
- Stations from filename char 15 (B Big Bear, C Cerro Tololo, L Learmonth, M Mauna Loa, T El Teide, U Udaipur), roughly balanced.
- `iscrowd` all 0; each annotation is a single polygon; areas 9–37,739 px² (raw) with only 51 "small" (<200) instances in the selected set.
- Categories: Left / Right / Unidentifiable (Ambiguous is unused after selection). Model is single-class (`NUM_CLASSES=1`).

## 4. Files that changed (all uncommitted)
New:
- `src/annotations.py` — loads COCO grouped by file stem; `select_annotations`, `load_gt_rles`.
- `src/split.py` — StratifiedGroupKFold (5 folds, fold 0 = val), 27-day groups, count-quantile strata, seed 42. Writes `splits/val_ids.json` once; `--force` to regenerate.
- `splits/val_ids.json` — **frozen: 566 train / 141 val. Never regenerate.**
- `src/validate_submission.py` — header, id format, test-image membership, bare-counts RLE, decode sample.
- `src/utils/{__init__,rle,seed,exp_log}.py` — RLE helpers, seeding, `experiments/log.csv` logger.
- `experiments/log.csv` — header only (schema from the plan).
- `tests/{conftest,test_metric,test_rle_and_submission,test_split}.py` — 14 tests.
- `notebooks/colab_foundation.ipynb` — 18-cell Colab notebook (clone → install → lock versions → tests → EDA → dry run → download results → Kaggle-notebook comparison steps).
- `HANDOVER.md` — this file.

Rewritten/modified:
- `src/config.py` — nested `CFG.paths / data / model / train / tiling / infer`; env vars `SOLAR_ROOT`, `SOLAR_DATA_DIR`, `SOLAR_OUTPUT_DIR`; no mkdir at import (call `CFG.paths.ensure_output_dirs()`).
- `src/evaluate.py` — dataset-pooled PQ/SQ/RQ, per-image mean PQ, semantic Dice, matched Dice, IoU percentiles, 1→N fragmentation and M→1 merge counters, PQ by size bin (<200, 200–2000, >2000), CLI (`--pred`, `--gt-vs-gt`).
- `src/dataset.py`, `src/train.py`, `src/inference.py`, `src/eda.py`, `src/visualize.py`, `run_pipeline.py` — updated to the new config/annotation layer; `run_pipeline.py` has `--step {eda,split,train,val,infer,all}`, `--dry-run`, `--name`, `--weights`.
- `requirements.txt` — loose pins with notes; exact cloud versions to come from `requirements-lock.txt`.
- `.gitignore` — added `submissions/*.csv`, `.pytest_cache/`.
- `CLAUDE.md` — added multi-annotator warning, human-ceiling PQ, frozen-split note, corrected instance stats.

Bugs fixed in old code: `total_mem` (should be `total_memory`) crash; inference kept all dense masks in RAM (now RLE immediately); train/test resize mismatch (old train used 640–800 multiscale, test 1024); old split was random per COCO entry (leaky); old PQ averaged per image instead of pooling.

## 5. Failed attempts / gotchas
- **WebFetch of the Kaggle self-evaluation notebook returned only the title** (JS-rendered). User must paste its PQ cell(s) and printed numbers.
- **Validator bug (fixed):** first version rejected `[` `]` in RLE strings; COCO counts legitimately contain them. Now only quotes/whitespace are forbidden (a size array is caught by the 2-column check).
- Detectron2 cannot be imported locally (IDE "missing import" errors are expected). The Detectron2 code paths ran fine in the Colab dry run, but only for 5 iterations on 8 images; a full run is still untested (periodic COCOEvaluator with string `image_id`s, checkpointing).
- **Notebook generation bug (fixed):** escape sequences like `\n` written through the Bash heredoc came out as real newlines in the .ipynb (f-string SyntaxError in the freeze cell). When generating notebook/code text via shell, avoid backslashes (use `chr(10)`) and compile-check every cell afterwards.
- Colab ships both `opencv-python` 5.0 and `opencv-python-headless` 4.14 (same `cv2`); harmless so far.
- `run_pipeline.py --dry-run` locally just prints that Detectron2 is required; that is correct behaviour.
- Windows console shows mojibake for "—" in some prints; cosmetic.
- Windows Python is 3.13 locally; Colab will differ.

## 6. Decisions made / open
- Baseline model: Detectron2 Mask R-CNN R50-FPN with anchors 8–128 (per plan). **Decided 2026-10-20 by the user: Mask R-CNN first; YOLOv8-seg later as a comparison.**
- `ANNOTATOR_POLICY="first"`; alternatives ("most", or training on all sets) to compare in Phase 2.
- The user explicitly wants Claude to commit/push only on request; no attribution decisions pending beyond the standard co-author trailer.
- User is new to segmentation; explanations so far have been plain-language (exam analogy: train = textbook, 141 val = practice test, 180 test = real exam scored by Kaggle).

## 7. Next steps
1. ~~Commit~~ (done, local). Push `Dev` when the user asks.
2. ~~Colab dry run~~ (done, exp001). ~~Lock file~~ (saved). Phase 0 and the split gate are ticked in `Project_Phase_Plan.md`.
3. **Phase 2 — Baseline model, metric alignment & first Kaggle submission (WAITING for user go-ahead; do not start on your own).**
   - ~~**2-0 Decide model**~~ — **DONE 2026-10-20:** Mask R-CNN R50-FPN first, YOLOv8-seg later as a comparison.
   - **2A — DONE 2026-10-20 (single-annotator case).** Official notebook PQ 0.3296 = ours 0.3296 on `output/metric_check/{gt,pred}.csv` (from `src/make_metric_check_csvs.py`; 62 val files, set 0 = GT, set 1 = pred). Findings: official GT ids are `<annotator>-<stem>_<i>` (COCO entry id), code splits on the first `-`; predictions use the plain stem. **Official code + analysis: see [`SelfEvaluationResult/FINDINGS.md`](SelfEvaluationResult/FINDINGS.md)** (verbatim code in `SelfEvaluationResult/official_metric.py`). Key: each annotator set is scored separately and pooled (a perfect copy of set 0 scores only 0.7394 officially on val), and matching is not one-to-one (duplicating predictions raises PQ 0.3296→0.3729 — loophole, not exploited). Next: `--official` mode in `evaluate.py`. Original plan below kept for reference.
     1. User opens https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook → Copy & Edit, runs it unchanged, records which GT/prediction files it reads and the PQ/SQ/RQ/Dice it prints.
     2. User pastes its PQ/matching cell(s) + printed numbers to Claude.
     3. Claude compares against `src/evaluate.py`: pooled vs per-image PQ, `IoU > 0.5` vs `>= 0.5`, matching of overlapping predictions, which Dice it uses (torchmetrics DiceScore settings), how it treats multi-annotator GT. Adjust `evaluate.py` + tests until the same inputs give the same PQ within ±0.001.
     4. If possible, run the official notebook on our own validation predictions (CSV from 2C) and compare numbers directly. Gate: tick the ±0.001 checkbox (Phase 1 section) in `Project_Phase_Plan.md`.
     - 2A can run in parallel with 2B; it must be done before trusting val PQ for model selection.
   - **2B/2C/2D SET UP 2026-10-20, NOT RUN (user: "don't execute yet").** Code + `notebooks/colab_phase2_baseline.ipynb` are ready; Colab clones `Dev` from GitHub, so the changes must be **committed and pushed** before the user runs it. Notebook flow: mount Drive (`MyDrive/solar_filament/output` via `SOLAR_OUTPUT_DIR`) → clone → install → tests (21) → `--step train` (resumable) → loss plot → `--step val` → `--step sweep` → val + infer with the best sweep cell → downloads. Checkpoints in `output/detectron2/<name>/`. `val_pq` in the log is now the **official** PQ; first-annotator PQ goes in `notes`. Untested on GPU: aspect-ratio RPN re-init (expect a "skip loading, shape mismatch" warning), resume, sweep. Original plan text below.
   - **2B Real baseline training on Colab:** Claude writes a Phase 2 Colab notebook (e.g. `notebooks/colab_phase2_baseline.ipynb`). Per plan: R50-FPN, anchors 8–128, aspect ratios `[0.2, 0.5, 1.0, 2.0, 5.0]` (not yet in config), 1024 input, LR 0.0025, batch 2, 5–7k iters, flips + brightness/contrast. Save checkpoints to Google Drive (Colab sessions die). Untested so far: periodic COCOEvaluator with string `image_id`s — consider setting `EVAL_PERIOD=0` and using our PQ instead.
   - **2C Validation:** `python run_pipeline.py --step val --weights <ckpt> --name baseline_r50` → val PQ/SQ/RQ, frag/merge, size-bin PQ; logged to `experiments/log.csv`. Consider a quick sweep of `SCORE_THRESHOLD` / `MIN_AREA` on val.
   - **2D Submission:** `python run_pipeline.py --step infer --weights <ckpt> --name baseline_r50` → `submissions/submission_baseline_r50.csv`, passes `validate_submission.py` (all 180 test images), user uploads to Kaggle, records public LB score in `lb_score` of the log row.
   - **Phase 2 exit gates:** baseline trained + evaluated on frozen val; first valid Kaggle submission scored with 0 errors; local metric aligned with the official notebook (2A).
4. Later (per plan): CLAHE + solar-disk masking, 2048 tiling with stitching, anti-fragmentation post-processing, Mask2Former/SAM 2, TTA/ensembling.

## 8. Standing rules (from CLAUDE.md)
- Never list/read files in `train_images/` or `test_images/`; never print the 48 MB annotations JSON (parse only in Python).
- Seeds 42; submissions go to `submissions/submission_<name>.csv`; validate before upload.

---

## 9. Update 2026-10-20 — Lecture-notes review (MCTA 4364 Machine Vision)
No code changed and Phase 2 is still **not started**. Only `Project_Phase_Plan.md` and this file were edited.

**Context:** the user added `lecturenotesMachineVision/` (course notes W00–W14, 28 notebooks; `notebooks/` inside it duplicates the `W*` folders). The competition was originally this subject's final project. All weeks were reviewed for relevance. Verdicts are reasoned from the EDA summary and sample images, **not measured** — the user declined a data-analysis run to save quota. Each added method is an ablation toggle that must earn its place on frozen-val PQ.

**Changes to `Project_Phase_Plan.md`:**
- **Phase 2:** added optional timeboxed (0.5 day, CPU) classical baseline (disk mask → background subtraction → adaptive threshold → closing → connected components → `MIN_AREA`) as a report comparison row; new optional exit-gate checkbox.
- **Phase 3:** limb-darkening normalisation made *conditional* (sample images already look radially flat; check a radial intensity profile first). Added shape-descriptor filter (elongation/solidity), off by default, only drop small + compact + very dark blobs (a global elongation cut would delete compact "Unidentifiable" GT). Added data note: smallest selected GT is 60 px², so `MIN_AREA` sweep 50–100 is safe.
- **Phase 5:** SAM 2 track now primarily box-prompt mask refinement of detector boxes (automatic mask generator demoted to secondary). Added U-Net (CE+Dice) + connected-components fallback if Mask2Former is No-Go/late. Go/No-Go item renumbered 3→4.
- **Phase 6:** watershed constriction splitting marked validation-gated (risk of creating 1→N fragmentation on naturally narrow spines).
- **Phase 7:** latency reported as median/p95; new per-station (B/C/L/M/T/U) and per-size-bin PQ breakdown; later items renumbered 5–6.
- **Phase 8:** error gallery (fragmentation, merge, faint, limb, FP) and W14 evaluation checklist added to report figures/review.
- **New Section 14 (appendix):** verdict table for every lecture method. Skipped: synthetic/copy-paste data (W13B), CLIP/VLM auto-labelling (W13A), optics/calibration/stitching/motion/3D (W01–02, 05B, 10, 12).

**Notes for next session:**
- Lecture notes also contain the course report/proposal template (W14) usable for the 30% qualitative score.
- Scratch tooling lives only in the Antigravity artifact dir (notebook text dump, an unrun GT-analysis script `analyze_gt.py` for aspect/solidity/limb-profile/watershed checks). It can be run later to validate the toggles above.
- Prior-session `Project_Phase_Plan.md` header still says "Current Date: Oct 18"; unchanged.
