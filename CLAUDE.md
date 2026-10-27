# CLAUDE.md — Solar Filament Segmentation Challenge 2026

## 0. Team session start (READ FIRST, every new session)
Three people work in this repo, each with their own Claude Code: **Kimee** (lead, Phases 0–5), **Mirul** (Phase 6
track M) and **Syaz** (Phase 6 track S). All are new to segmentation: explain in plain language, short sentences.

1. **Find out who you are working with.** If the first message doesn't say, ask: "Are you Kimee, Mirul or Syaz?"
2. **Read before answering:** `HANDOVER.md` (whole file), `PHASE6_GUIDE.md` (plain-language task explanations — use its
   wording when explaining), then that person's part of `Project_Phase_Plan.md` → Phase 6
   (Mirul §6.2, Syaz §6.3, shared rules §6.4), and their subsection of `COMPLETED_TASKS.md` (Phase 6).
3. **First session for Mirul or Syaz (onboarding):** explain, in this order and briefly:
   (a) the competition and the metric in two or three sentences; (b) the timeline and what Kimee already did
   (`HANDOVER.md` §0.5 and §2, with the current best scores); (c) why Phase 6 exists (gains below the noise);
   (d) their own tasks, in order, with what "done" means and roughly how long each takes; (e) how they work with
   the others (git rules, group chat, experiment-ID range, result folder). Then check their setup with them step by
   step (`HANDOVER.md` §0.3: clone, `setup_local.ps1`/`.sh`, Colab T4, team Drive shortcut, Kaggle team) and propose
   their first task.
4. **Later sessions:** run `git pull origin Dev` only if the person agrees, then say in 3–5 lines what changed since
   their last session (both Live status blocks in `HANDOVER.md` §2a, new rows in `experiments/log.csv`,
   `COMPLETED_TASKS.md`) and what their next task is.
5. **MANDATORY — keep `HANDOVER.md` current (`HANDOVER.md` §0.7).** After every finished task or step, scored Colab
   run, Kaggle upload or changed decision, and before every push with results, update the member's own **Live status**
   block in §2a (and the results block / §2 / §6 when relevant) without being asked. A task is not done until
   HANDOVER is updated. A fresh session must be able to onboard anyone from the repo alone.
   **Every session, no exceptions (Kimee's rule):**
   - **Start:** compare the member's Live status block with `experiments/log.csv`, `COMPLETED_TASKS.md` and
     `git log`. If it is out of date (results, runs or uploads not in it), fix it first and tell the member.
   - **End:** before the session ends (the member says bye/done/stop, or the work pauses), update the Live status
     block even if nothing finished: set the date, what was done, what is half-done (e.g. a Colab run still
     training: run name, where its checkpoints are), next step, blockers. Then ask the member to commit and push it
     ("Mirul: update HANDOVER") and post in the group chat. An update that is not pushed is invisible to the others.

**Working rules (all members):**
- **Explain first, run after a "go".** Before a new task or anything that runs code on Colab, explain the plan in plain
  language and wait for an explicit go. A "yes" to a question is not a go for the next task.
- **Don't re-run past work** — check §2b files first and reuse recorded numbers.
- **Keep helper tools simple** — small scripts, no extra features nobody asked for.
- **Git:** never commit or push unless the person asks. Always `git pull origin Dev` before a push; after pushing,
  remind them to post "pushed new version: <what>" in the group chat. Never force-push. On a conflict, help merge;
  never drop the other person's changes.
- **Stay in your lane:** write only to the member's own files (`Result Phase 6/<Name>/`, their notebook/scripts, their
  subsection of `COMPLETED_TASKS.md`, their rows in logs). Experiment IDs: Mirul exp101+, Syaz exp201+, Kimee exp014+
  (pass `exp_id=` explicitly to `log_experiment`).
- **Team Drive folder** (Kimee's `solar_filament`, shared by link): **never write the link into any repo file** (the
  repo goes public). Never modify or delete existing files there (Kimee's trained 2048 model checkpoints); write
  only inside `Phase 6/<Name>/` (checkpoints in `Phase 6/<Name>/output/`) or `external/`; run names start with the
  member's name (`HANDOVER.md` §0.6).
- Only Mirul uploads to Kaggle in Phase 6 (≤ 2/day, only for candidates above the M1 noise threshold).
- Colab clones branch `Dev`: code must be pushed before a notebook uses it. Mirul and Syaz have **T4 GPUs only**
  (15 GB): no 2048 training there.

## 1. Project & Objective
- **Task**: Instance segmentation of solar filaments from ground-based GONG H-Alpha images.
- **Goal**: Generate precise masks for individual filaments, capturing fine barbs while avoiding false positives and fragmentation.
- **Deadline**: Nov 15, 2026 (code freeze Nov 11). Best so far: val official PQ 0.4366, public LB 0.38 (exp010). Phase 6 target: LB 0.40–0.42.

## 2. Dataset Architecture
```
filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/
├── train/
│   ├── MAGFiLO_1.0_Annotations_kaggle2026_train.json  # COCO polygon format (8,199 instances, spines included)
│   └── train_images/                                 # 707 JPEGs (2048x2048)
└── test/
    └── test_images/                                  # 180 JPEGs (2048x2048, unannotated)
```
- **Image Specs**: Fixed 2048 × 2048 px single-channel/RGB JPEG.
- **Instance Properties**: Areas range from 9 px² to 37,739 px² (mean ~2,120 px²).
- **Multi-annotator quirk (CRITICAL)**: JSON has 1,154 image entries for 707 files; 296 files were annotated 2–3× independently (each set = separate entry). Never merge sets or treat entries as images — use `src/annotations.py` (one set per file, policy `first`; ~7.4 filaments/file). Training since Phase 3 uses `--label-policy all` (each annotator set = its own sample); evaluation always uses all sets.
- **Inter-annotator PQ ≈ 0.336** (human ceiling; see `output/eda/eda_summary.json`).
- **Frozen split**: `splits/val_ids.json` (566/141, 27-day time blocks). Never regenerate.
- **Token Saver Guardrails (STRICT)**:
  - NEVER list or read files in `train_images/` or `test_images/`. Treat image files as opaque paths loaded only by Python `DataLoader`.
  - NEVER print or read the raw `MAGFiLO_1.0_Annotations_kaggle2026_train.json` (48 MB) into context; parse it only via Python scripts.

## 2b. Check Past Results Before Running Anything (STRICT — saves time and tokens)
Read these first; never re-run work they already record unless the code it depends on has changed:
- `COMPLETED_TASKS.md` — every finished task by phase (Phase 0, 1 and metric alignment are DONE).
- `SelfEvaluationResult/FINDINGS.md` — official Kaggle metric analysis; `official_metric.py` is the organisers' code (verbatim).
- `HANDOVER.md` — current state, decisions, next steps, gotchas.
- `experiments/log.csv` — every training/validation run and its scores.
- `output/eda/eda_summary.json` — EDA numbers (do not recompute).
- `Solar Filament Overview.txt` — saved Kaggle overview page (rules, rubric, deadlines); Kaggle pages cannot be fetched. Summary in `HANDOVER.md` §2b.

Rules:
- Do NOT re-run past phases (EDA, split, dry run, Kaggle metric check). The frozen split and metric values are final.
- Run only the tests covering code you changed (e.g. `pytest tests/test_official_metric.py`); the full suite only before a commit.
- Known values to reuse, not recompute: inter-annotator PQ 0.336; metric check 0.3296 (Kaggle = ours); a copy of one annotator scores 0.7394 official PQ on val.
- After any new run, record its result in `COMPLETED_TASKS.md` (and `experiments/log.csv` / `FINDINGS.md` where relevant).
- **Model selection uses `official_PQ`** (all annotator sets, Kaggle matching), not the single-annotator `PQ`.

## 3. Evaluation Metric: Panoptic Quality (PQ)
- $\text{PQ} = \text{SQ} \times \text{RQ}$
  - $\text{SQ} = \frac{\sum_{(p, g) \in \text{TP}} \text{IoU}(p, g)}{|\text{TP}|}$ (mean IoU of true positives where $\text{IoU} > 0.5$)
  - $\text{RQ} = \frac{|\text{TP}|}{|\text{TP}| + 0.5|\text{FP}| + 0.5|\text{FN}|}$
- **Critical Failure Modes to Avoid**:
  - **Fragmentation**: Splitting 1 filament into $k$ parts creates 1 TP + $(k-1)$ False Positives, destroying RQ.
  - **Over-merging**: Merging 2 separate filaments into 1 creates False Negatives.
  - **Spurious Blobs**: Tiny noise clusters (< 30–50 px²) trigger massive FP penalties.

## 4. Submission Specifications
- Format: Single CSV with header `filament_id,segmentation_rle`.
- `filament_id`: `<image_id>_<index>` (e.g., `20150125172714Mh_1`). Image ID is filename without extension.
- `segmentation_rle`: COCO RLE counts string (via `pycocotools`).
  - **DO NOT** wrap strings in quotes.
  - **DO NOT** output the image size array; only the counts string. Fixed dimensions are 2048 × 2048.

```python
# Standard RLE generation
from pycocotools import mask as coco_mask
import numpy as np

def mask_to_rle(binary_mask_2048x2048: np.ndarray) -> str:
    rle = coco_mask.encode(np.asfortranarray(binary_mask_2048x2048.astype(np.uint8)))
    return rle['counts'].decode('utf-8')
```

## 5. Pipeline & Modeling Strategy
> **Original plan from Oct 18 — partly outdated.** What was actually tested and kept is in `HANDOVER.md` §2 and
> `experiments/ablation_table.md`: CLAHE + disk masking gave no gain; native 2048 (no tiles) replaced tiling;
> Mask2Former and SAM 2 were rejected; the pipeline is Mask R-CNN (X101-FPN with 56×56 mask head + R50-FPN) ensemble.

1. **Preprocessing**:
   - **Solar Disk Masking**: Apply Hough circle transform to detect the solar disk; zero out background/corners to eliminate corner artifacts.
   - **Enhancement**: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to reveal thin filament barbs.
2. **Inference / Resolution Handling**:
   - Full 2048×2048 causes GPU OOM on dense transformers.
   - Use overlapping sliding window tiles (e.g., 1024×1024 with 256px overlap or 512×512 with 128px overlap).
   - Stitch tile masks back to global coordinate space using Non-Maximum Suppression (NMS, IoU ~0.3–0.4).
3. **Model Selection**:
   - *Baseline*: YOLOv8-seg or Mask R-CNN (ResNet-50 FPN with small anchor sizes: 8, 16, 32, 64, 128).
   - *Advanced*: Mask2Former (Swin-T/S backbone) or fine-tuned SAM 2.
4. **Post-Processing (PQ Boosters)**:
   - **Morphological Closing**: Kernel (3×3 or 5×5) to reconnect broken filament spines/barbs.
   - **Area Thresholding**: Drop any predicted instance with area < 30–50 px² to reduce FP.

## 6. Coding & Workflow Rules
- **Environment**: Python 3.10+, PyTorch with CUDA, `pycocotools`, `albumentations`, `opencv-python`.
- **Reproducibility**: Always set random seeds (`torch.manual_seed(42)`, `np.random.seed(42)`).
- **Validation**: 80/20 train/val split (stratified by filament count or time block). Always evaluate locally using official PQ before submitting.
- **Code Cleanliness**: Keep scripts modular (separate `dataset.py`, `models/`, `utils/rle.py`, `train.py`, `predict.py`).
- **Submission Output**: Save predictions to `submissions/submission_<model_name>.csv` and validate row counts and RLE strings before upload.
