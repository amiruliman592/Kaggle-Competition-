# Phase 6 guide — what you are doing and why (Mirul, Syaz)

This is the plain-language version. The exact task list (inputs, file names, "done when") is in
`Project_Phase_Plan.md` → Phase 6, and the current status is in `HANDOVER.md` §2a. Your Claude Code session has read
all of this; ask it anything ("explain M1 again", "what is IoU?").

## 1. The project in two minutes
- **Data:** 887 photos of the Sun (2048 × 2048 pixels) taken in red "H-alpha" light by the GONG telescopes. Solar
  **filaments** are the long dark threads on the disk. 707 photos come with answers (outlines drawn by experts),
  180 are the hidden test set.
- **Task:** for every filament, draw its exact outline (a **mask**). One CSV row per filament.
- **Our model:** **Mask R-CNN**. It first puts a box around each thing it thinks is a filament, with a confidence
  **score** (0–1), then paints the filament's pixels inside the box. We use two versions: **R50** (smaller, faster)
  and **X101** (bigger). "2048" in a name = trained on full-size images; "1024" = on half-size images (faster).
- **Training** = showing the model the 707 labelled photos thousands of times so it learns. The result is a
  **checkpoint** file (`model_final.pth`) — the model's learned "memory". Kimee's best checkpoints live in the team
  Drive folder; they cost hours of paid A100 GPU time, so we only read them.
- **Ensemble** = letting two models vote: a filament both models find is kept with a high score. Our best submission
  is an ensemble of X101 + R50.

## 2. How we are scored
- A predicted outline is a **hit** if it overlaps a true filament by more than half (**IoU > 0.5**; IoU = overlap
  area ÷ combined area).
- **PQ (Panoptic Quality)** = (how well the hits fit) × (hits ÷ (hits + half the misses + half the false alarms)).
  Splitting one filament into pieces or merging two into one both hurt.
- **Dice** is a second score the judges also look at (pixel overlap).
- **Human level:** two experts outlining the same photo agree with each other only at PQ **0.336** — the task is
  genuinely fuzzy, so all scores look low. We are at **0.44 on val** and **0.38 on the Kaggle leaderboard**.
- **Val vs leaderboard:** **val** = 141 of the labelled photos we never train on; we score them on the laptop, as
  often as we like. **Leaderboard (LB)** = Kaggle's score on ~90 hidden test photos, shown with only 2 decimals,
  5 uploads per day for the whole team. Val is always ~0.05 higher than LB; what matters is that both go up.

## 3. Why Phase 6 exists
Think of judging a basketball player on 10 shots: 7/10 vs 6/10 does not prove the first is better. Our last
"improvements" were +0.002 to +0.01 PQ, which is the same kind of luck: two of them looked better on val and then
scored *lower* on the LB. So Phase 6 does two things:
1. **Measure the luck ("noise")** so we know how big a gain must be to be real (Mirul, M1; Syaz, S1).
2. **Try bigger changes** that should give clearly more than that: show each image to the model from several angles
   (Mirul, M2), train better (Syaz, S2–S4), and give it more photos to learn from (Mirul M3 + Syaz S5).

Target: LB 0.40–0.42. When both of you are done, Kimee trains the final big models with the best recipe you found.

## 4. Words you will see
| Word | Meaning |
|---|---|
| raw predictions | every mask the model drew, with its score, before any cleaning (`raw_preds_*.json.gz`) |
| threshold (thr) | masks with a score below it are thrown away (we use 0.6–0.7) |
| post-processing | cleaning after prediction: close small gaps, drop tiny blobs, remove overlaps |
| iteration / step | one training update on 2 images; 6,000 steps ≈ 13 passes over the data |
| seed | the starting point of the randomness in training; a different seed = a slightly different model |
| augmentation | showing training photos changed (flipped, rotated, brighter) so the model learns more from them |
| TTA | test-time augmentation: predict on flipped copies too, then let the copies vote |
| pseudo-labels | outlines drawn by our own model on new, unlabelled photos, used as extra training answers |
| bootstrap | re-scoring many random re-draws of the 141 val photos to see how much a score wobbles |
| exp ID | the row number of a run in `experiments/log.csv` (Mirul exp101+, Syaz exp201+) |

## 5. Mirul — track M
**M1. How big is the luck, and where do we lose points? (laptop only, ~2–3 days, first)**
- *What:* Claude writes two small scripts. `bootstrap.py` re-scores our best runs on ~2,000 random re-draws of
  the val photos and says "a difference smaller than X is luck" — for val and for a simulated leaderboard.
  `error_budget.py` sorts every mistake of the best model into boxes: *almost right* (outline a bit off),
  *found but thrown away* (score too low), *never found*, *false alarm that another expert did draw*, *real false
  alarm* — per filament size and per telescope station.
- *Why:* everyone uses X to accept or reject results, and the error boxes tell us which fix is worth doing.
- *You:* run the scripts on the laptop (no Colab), read the result with Claude, post X in the group chat.

**M2. Let the models look three more times (Colab T4, ~1–2 h, no training)**
- *What:* `notebooks/colab_phase6_mirul.ipynb` runs Kimee's two best models on every val/test photo mirrored
  left-right, upside-down and both. Back on the laptop, Claude merges the 4 views per model like an ensemble and
  scores it.
- *Why:* a filament found in all 4 views is almost surely real; one found once is probably noise. This often gives a
  free gain without any training.
- *You:* run the notebook, download 12 files into `Result Phase 6/Mirul/`, tell Claude. If the gain is bigger than
  X, you upload the new CSV to Kaggle (max 2 uploads/day).

**M3. More photos for training (Colab + laptop, ~3 days)**
- *What:* download ~3,000 extra GONG H-alpha photos from the public NSO archive (the competition host said on the
  Kaggle forum this is allowed if no photo is within ±1 hour of a test photo; we skip whole test **and** val days to
  be safe). Our best model then draws outlines on them (pseudo-labels); only confident outlines are kept.
- *Why:* the model has only 707 labelled photos. More examples, even with machine-made answers, often help.
- *You:* run the download and labelling with Claude's help; put photos + labels in the team Drive `external/`; tell
  Syaz they are ready (Syaz trains on them in S5).

**M4 (if time).** A table of scores per telescope station and filament size, needed for the report.

## 6. Syaz — track S
All your runs train the smaller R50 model on half-size (1024) photos — that fits the free T4 GPU and takes 1–5 h.
Each run changes **one thing**, and is compared with Kimee's same-size run **exp006 (val PQ 0.3979)**.

**S1. Train exp006 again with another seed (~1.3 h) — notebook `colab_phase6_syaz.ipynb`**
- *What:* the exact same training, only the random starting point differs.
- *Why:* tells us how much the score moves by pure chance when re-training. Together with Mirul's X, this is the bar
  every later run must clear. It is also your "learn the workflow" run: no new code needed.

**S2. Train 3× longer (~4 h) — same notebook**
- *What:* 18,000 steps instead of 6,000.
- *Why:* 6,000 steps is short for this model; training longer is the simplest possible improvement. The winner of
  S2 (6k or 18k) is the length used for S3–S5.

**S3. Rotations + size changes (~4 h)** — Claude adds the code, then a new notebook cell.
- *What:* during training, photos are also turned by 90°/180°/270° and zoomed in/out a little (±20%).
- *Why:* filaments appear at all angles and sizes; the model now only sees flips. Not guaranteed: the Sun's north is
  always up in these photos, so we test it instead of assuming it.

**S4. Copy-paste (~4 h)** — dropped first if time runs short.
- *What:* during training, real filaments cut out of other photos are pasted onto empty spots of the solar disk.
- *Why:* more filaments per photo = more to learn from. A known trick from research on instance segmentation.

**S5. Train with Mirul's extra photos (~5 h)**
- *What:* the best recipe so far, plus Mirul's ~3,000 pseudo-labelled photos.
- *Why:* tests whether more (machine-labelled) data beats a better recipe.

**At the end:** you write the winning recipe (exact settings) into `HANDOVER.md` §2a so Kimee can use it for the
final big run.

## 7. A normal working day
1. Open the repo folder in VS Code → Claude Code → *"I'm Syaz, let's continue."*
2. Claude pulls (if you agree), tells you what changed and what is next, and explains the next step.
3. You say **"go"**. Claude writes the code + a small test, and asks you to push (`git pull origin Dev`, then
   `git push origin Dev`) and post "pushed new version: …" in the group chat.
4. You run the Colab notebook (it always pulls the newest code), download the result files into
   `Result Phase 6/<your name>/`.
5. Claude scores them on the laptop, logs the numbers, and **updates your Live status in `HANDOVER.md`** (mandatory).
   You push again and tell the group.

**Stuck?** Paste the error to Claude. If Colab is out of GPU time, try again later or ask the other member to run it;
Kaggle Notebooks (free ~30 GPU h/week) are the backup.
