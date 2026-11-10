# Solar Filament Instance Segmentation — Kaggle 2026

Instance segmentation of solar filaments in 2048 × 2048 GONG H-alpha images
(Solar Filament Segmentation Challenge 2026, MAGFiLO 1.0 data). Metric: Panoptic Quality (PQ = SQ × RQ).

**Final submission: validation PQ 0.4480, Dice 0.6520 · public leaderboard 0.39.**
For reference, two human annotators of the same image agree at PQ ≈ 0.336.

Team: Kimee (lead), Mirul, Syaz.

## Guide for reviewers: where to find what

Everything needed to reproduce the submission is in this repository: code, data, the frozen validation split and
the **model weights**. Nothing is downloaded from anywhere else except Python packages.

| If you want to… | Open | Notes |
|---|---|---|
| **See the whole pipeline run, with results** | [`notebooks/full_pipeline.ipynb`](notebooks/full_pipeline.ipynb) | Start here. The notebook is saved with the outputs of our own clean-clone run, so you can read the results without running it. To run it yourself: Colab, GPU runtime, "Run all" (about 45 minutes). |
| **Read the final pipeline code** | [`src/final_pipeline.py`](src/final_pipeline.py) | One file, about 200 lines: 2 models × 4 flipped views → fusion → post-processing → CSV. The header comment lists every step and setting. |
| **Find the model weights** | [`weights/`](weights/) | 2 models, stored as parts under 100 MB with a SHA-256 manifest. [`src/weights_io.py`](src/weights_io.py) joins and checks them. |
| **Find the final submission file** | [`results/phase7/final_run/submission_exp101.csv`](results/phase7/final_run/) | The submitted CSV (1,256 masks). `submission_final.csv` next to it is what the clean-clone run produced. |
| **Check that the result reproduces** | [`results/phase7/final_run/`](results/phase7/final_run/) | Reports of the clean-clone run: validation score, latency, GPU memory. Summary in the next section. |
| **See how the models were trained** | [`run_pipeline.py`](run_pipeline.py), [`src/train.py`](src/train.py), [`src/config.py`](src/config.py) | Exact commands are under "Training" below. |
| **See what we tried and what each step gained** | [`experiments/ablation_table.md`](experiments/ablation_table.md), [`experiments/log.csv`](experiments/log.csv) | Every run with its validation score, including the rejected ideas. |
| **Check our metric implementation** | [`src/evaluate.py`](src/evaluate.py), [`SelfEvaluationResult/`](SelfEvaluationResult/) | Matches the organisers' self-evaluation notebook exactly (0.3296 = 0.3296). |

Folders that are **not** part of the final pipeline, kept as a record of the work:
[`notebooks/development/`](notebooks/development/) (the Colab notebooks of each project phase),
[`results/`](results/) (saved predictions of every run), [`docs/`](docs/) (our team's working notes and plans).

## Reproduce the final submission

**Colab (recommended):** open [`notebooks/full_pipeline.ipynb`](notebooks/full_pipeline.ipynb), choose a GPU
runtime, Run all. It clones this repository, installs Detectron2, joins and checks the weights, predicts the 180 test
images, writes and validates the submission CSV, compares it with the submitted predictions, scores the validation
split and shows example masks. It does not use Google Drive or any other storage.

**Any Linux machine with a CUDA GPU (peak GPU memory use is 2.9 GB):**
```bash
git clone -b Dev https://github.com/amiruliman592/Kaggle-Competition-.git && cd Kaggle-Competition-
pip install -r requirements.txt
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git@1e3e13bbf607b54f62205c4c33922521822fb298'
python src/weights_io.py unpack --dir weights                       # join parts, check SHA-256
python src/final_pipeline.py predict --split test --out submissions/submission_final.csv
python src/final_pipeline.py predict --split val                    # official PQ + Dice on the validation split
```

**What our clean-clone run gave** (fresh Colab session, NVIDIA L4, 2026-11-10):

| | Clean clone | Submitted run |
|---|---|---|
| Validation PQ / Dice | 0.4480 / 0.6520 | 0.4480 / 0.6520 |
| Validation TP / FP / FN | 1083 / 495 / 595 | 1083 / 495 / 595 |
| Test masks | 1,255 | 1,256 |

1,255 of the 1,256 submitted test masks are found again, 1,249 of them at IoU > 0.99. One mask falls just below the
score threshold because of GPU rounding (the original predictions were made on a T4 and an A100).
Speed: 6.0 s per image (median; 8.2 s at the 95th percentile) for the whole pipeline, 18 minutes for the 180 test
images, 2.9 GB peak GPU memory.

Package versions: `requirements.txt` gives the allowed ranges; `requirements-lock.txt` gives the exact versions of our
training runs (Python 3.13, torch 2.11 + CUDA 13.0, Detectron2 commit `1e3e13b`). The clean-clone run used Colab's
newer default (torch 2.14.1) with the same Detectron2 commit and gave the same validation score.

## The final pipeline

| Step | What |
|---|---|
| Input | The test image only, at native 2048 × 2048. No tiling, no pre-processing. |
| Models | Mask R-CNN **X101-FPN with a 56 × 56 mask head** and Mask R-CNN **R50-FPN** (Detectron2, COCO-pretrained), both trained at 2048. |
| Test-time augmentation | Each model predicts the original image and its horizontal, vertical and double flip; masks are flipped back. |
| Fusion 1 (per model) | The 4 views are merged: masks with IoU ≥ 0.5 are one filament, mask = union, score = mean over the views. |
| Fusion 2 (models) | The 2 models are merged: IoU ≥ 0.3, mask = intersection, score = mean over the models. |
| Post-processing | Closing (5 px), score ≥ 0.6, area ≥ 50 px², overlapping masks removed (the higher score keeps shared pixels; a mask more than 20 % covered is dropped). |
| Output | `filament_id,segmentation_rle` (COCO RLE counts), checked by [`src/validate_submission.py`](src/validate_submission.py). |

### Training

Both models were trained on Google Colab (A100) with the same recipe: every annotator set of the 566 training files
is its own sample (921 samples), 6,000 iterations, batch size 2, learning rate 0.0025, horizontal and vertical flips
plus brightness and contrast changes, anchors of 8–128 px with aspect ratios 0.2–5, seed 42.

```bash
python run_pipeline.py --step train --name x101_2048_m28 --backbone x101 --resolution 2048 --mask-res 28 --label-policy all
python run_pipeline.py --step train --name r50_2048 --resolution 2048 --label-policy all
```
The notebooks that ran these commands are `colab_phase5_a100.ipynb` and `colab_phase4_native.ipynb` in
[`notebooks/development/`](notebooks/development/).

## How we got here

Validation PQ is the official metric on 141 held-out files; LB is the public leaderboard.

| Step | Val PQ | LB |
|---|---|---|
| Mask R-CNN R50 at 1024 px (baseline) | 0.3905 | 0.34 |
| + overlap rule, every annotator set as a training sample | 0.3979 | 0.35 |
| + native 2048 resolution (R50 / X101) | 0.4231 / 0.4264 | 0.36 / 0.37 |
| + ensemble of X101 and R50 | 0.4366 | 0.38 |
| + finer mask head, flip test-time augmentation, two-stage fusion (**final**) | **0.4480** | **0.39** |

Tried and rejected: CLAHE with solar-disk masking, SAM 2 mask refinement, Mask2Former (Swin-S), a three times longer
schedule, 90° rotations with zoom, copy-paste of filaments, self-training on 2,098 external GONG images, a third
ensemble member, and a learned re-scoring of the masks. A paired bootstrap on the validation split showed that gains
below about 0.010 PQ are noise, so only larger gains were kept. All numbers are in
[`experiments/ablation_table.md`](experiments/ablation_table.md).

## Validation

- Frozen split [`splits/val_ids.json`](splits/val_ids.json): 566 training and 141 validation files, split by
  27-day time blocks so that no day appears on both sides.
- [`src/evaluate.py`](src/evaluate.py) implements the official metric: every annotator set is scored against the
  predictions and pooled.
- Dice: torchmetrics `DiceScore`, samplewise.
- 296 of the 707 training files were annotated 2–3 times; [`src/annotations.py`](src/annotations.py) handles the
  repeated entries.

## Repository layout

```
README.md                    this guide
LICENSE                      MIT licence for our code and trained weights
requirements.txt             Python packages (requirements-lock.txt = exact versions of our runs)
run_pipeline.py              command-line entry for training and single-model prediction
setup_local.ps1 / .sh        one-command CPU setup for a laptop (scoring, fusion and tests need no GPU)

notebooks/
  full_pipeline.ipynb        FINAL: the whole pipeline from a clean clone, saved with outputs
  development/               Colab notebooks of each project phase (history)

src/
  final_pipeline.py          FINAL: 2 models x 4 views -> fusion -> post-processing -> CSV
  weights_io.py              FINAL: join and check the weights stored in weights/
  inference.py               prediction of one model, including flipped views
  ensemble.py                fusion of masks from several views or models
  postprocess.py             closing, score and area cut, overlap removal
  validate_submission.py     submission format check
  utils/                     RLE encoding, flips, seeding, experiment log
  config.py, dataset.py, train.py, augment.py, annotations.py, split.py, preprocess.py   data and training
  evaluate.py, bootstrap.py, error_budget.py, ablate_postprocess.py, sweep.py  scoring and analysis
  sam_refine.py, m2f.py, copy_paste.py, pseudo.py, rescore.py, external/   rejected experiments
  eda.py, viewer.py, visualize.py, fix_overlaps.py, submit_from_raw.py, make_metric_check_csvs.py   small tools

weights/                     final model weights (parts + manifest.json)
filament-segmentation-2026/  competition data
splits/                      frozen train / validation split
results/                     saved predictions and scores of every run, by phase (see results/README.md)
experiments/                 experiment log and ablation tables
SelfEvaluationResult/        the organisers' metric code and our comparison with it
tests/                       test suite (python -m pytest tests)
docs/                        team working notes: plan, hand-over notes, task log, competition overview
external/                    list of the external GONG images tried for self-training (not used in the final model)
CLAUDE.md                    instructions for the AI coding assistant the team used during development
```

## Rules we followed

Only the test images are used at inference: no time or station from the file name, no neighbouring frames. No
external labels were used (no MAGFiLO masks, no filament catalogues). External unlabelled GONG images were tried for
self-training only, as allowed by the host and with no day within ±1 of a test image; they are **not** used in the
final model.

## Licence

Our code and our trained model weights are released under the [MIT licence](LICENSE). This does not cover material
that belongs to others and is included here only so that the result can be reproduced:
the competition data in `filament-segmentation-2026/` (MAGFiLO 1.0, under the competition's terms) and the
organisers' metric code in `SelfEvaluationResult/official_metric.py`. The models are built with
[Detectron2](https://github.com/facebookresearch/detectron2) (Apache 2.0) and start from its COCO-pretrained weights.
