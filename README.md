# Solar Filament Instance Segmentation — Kaggle 2026

Instance segmentation of solar filaments in 2048 × 2048 GONG H-alpha images
(Solar Filament Segmentation Challenge 2026, MAGFiLO 1.0 data). Metric: Panoptic Quality (PQ = SQ × RQ).

**Final submission (exp101): val official PQ 0.4480, Dice 0.6520 · public LB 0.39.**
For reference, two human annotators of the same image agree at PQ ≈ 0.336.

## Reproduce the final submission

Everything is in this repo: code, data, frozen split and **model weights** (`weights/`, split into < 100 MB parts).

**Colab (recommended):** open [`notebooks/full_pipeline.ipynb`](notebooks/full_pipeline.ipynb), choose a GPU runtime
(T4 is enough), Run all. It clones the repo, installs Detectron2, joins and checks the weights (SHA-256), predicts
the 180 test images, writes and validates `submissions/submission_final.csv`, compares it with the submitted
predictions, scores the val split and shows example masks.

**Any Linux machine with a CUDA GPU:**
```bash
git clone <this repo> && cd <repo>
pip install -r requirements.txt
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git@1e3e13bbf607b54f62205c4c33922521822fb298'
python src/weights_io.py unpack --dir weights                       # join parts, check SHA-256
python src/final_pipeline.py predict --split test --out submissions/submission_final.csv
python src/final_pipeline.py predict --split val                    # official PQ + Dice on the frozen val split
```
Measured on a fresh Colab clone (NVIDIA L4): val official PQ 0.4480 (same as the submitted run), 6.0 s per image
(median; p95 8.2 s; 8 forward passes + fusion + post-processing), 180 test images in 18 min, peak GPU memory 2.9 GB.
Test predictions match the submitted ones up to GPU rounding (1,255 of 1,256 masks, see
`experiments/ablation_table.md`, last section).

Exact package versions of our training runs: `requirements-lock.txt` (Python 3.13, torch 2.11 + CUDA 13.0, Detectron2 `1e3e13b`).
The clean-clone run above used Colab's newer default (torch 2.14.1 + CUDA 13.0) with the same Detectron2 commit and gave the same val score.

## Final pipeline (`src/final_pipeline.py`)

| Step | What |
|---|---|
| Input | Test image only, native 2048 × 2048, no tiling, no pre-processing |
| Models | Mask R-CNN **X101-FPN with a 56 × 56 mask head** + Mask R-CNN **R50-FPN**, both trained at 2048 (Detectron2, COCO init) |
| Training | Every annotator set of the 566 train files is its own sample (921 samples); 6,000 its, batch 2, LR 0.0025, h/v flips + brightness/contrast; anchors 8–128 px, aspect ratios 0.2–5 |
| Test-time augmentation | Each model predicts the original + horizontal, vertical and double flip (masks flipped back) |
| Fusion 1 (per model) | 4 views: IoU ≥ 0.5 = same filament, mask = union, score = mean over views |
| Fusion 2 (models) | IoU ≥ 0.3, mask = intersection, score = mean over the 2 models |
| Post-processing | Closing (5 px), score ≥ 0.6, area ≥ 50 px², overlaps removed (higher score wins; > 20 % covered → dropped) |
| Output | `filament_id,segmentation_rle` (COCO RLE counts), checked by `src/validate_submission.py` |

## How we got here (val official PQ on 141 held-out files; public LB)

| Step | Val PQ | LB |
|---|---|---|
| Mask R-CNN R50, 1024 px (baseline) | 0.3905 | 0.34 |
| + overlap rule, all annotator sets as training samples | 0.3979 | 0.35 |
| + native 2048 resolution (R50 / X101) | 0.4231 / 0.4264 | 0.36 / 0.37 |
| + ensemble X101 + R50 | 0.4366 | 0.38 |
| + finer mask head, flip TTA, two-stage fusion (**final**) | **0.4480** | **0.39** |

Tried and rejected (all numbers in [`experiments/ablation_table.md`](experiments/ablation_table.md) and
[`experiments/log.csv`](experiments/log.csv)): CLAHE + solar-disk masking, SAM 2 mask refinement, Mask2Former
(Swin-S), longer schedule, 90° rotations + zoom, copy-paste of filaments, self-training on 2,098 external GONG
images, a third ensemble member, learned re-scoring. A paired bootstrap on val showed that gains below ~0.010 PQ
are noise, so only larger gains were kept.

## Validation

- Frozen split [`splits/val_ids.json`](splits/val_ids.json): 566 / 141 files in 27-day time blocks (no same-day leakage).
- `src/evaluate.py` implements the official metric (every annotator set scored and pooled); it matches the
  organisers' self-evaluation notebook exactly (0.3296 = 0.3296, `SelfEvaluationResult/FINDINGS.md`).
- Dice: torchmetrics `DiceScore`, samplewise.
- 296 of the 707 training files were annotated 2–3 times; `src/annotations.py` handles the duplicate entries.

## Repository layout

```
src/final_pipeline.py      final submission pipeline (GPU) + submission comparison
src/weights_io.py          weights stored in the repo (pack / unpack + SHA-256)
src/train.py, dataset.py   Detectron2 training (config: src/config.py, CLI: run_pipeline.py)
src/inference.py           prediction, incl. flipped views
src/ensemble.py            mask fusion of models / views (CPU)
src/postprocess.py         closing, score / area cut, overlap removal (CPU)
src/evaluate.py            official PQ + Dice
src/bootstrap.py           paired bootstrap (noise threshold)
src/error_budget.py        where PQ is lost (by error type, size, station)
weights/                   final model weights (parts + manifest.json)
filament-segmentation-2026/  competition data
splits/                    frozen train/val split
notebooks/                 Colab notebooks of every phase; full_pipeline.ipynb = final
Result */                  saved raw predictions of every run (scoring and fusion run on CPU from these)
experiments/               experiment log + ablation tables
tests/                     pytest suite (python -m pytest tests)
```

## Rules we followed

Only the test images are used at inference (no file-name time or station, no neighbouring frames). No external
labels (MAGFiLO masks, filament catalogues). External unlabelled GONG images were tried for self-training only
(host-approved, no day within ±1 of a test image) and are **not** used in the final model.

## Team

Kimee (lead), Mirul, Syaz.
