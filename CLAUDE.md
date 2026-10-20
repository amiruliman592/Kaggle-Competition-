# CLAUDE.md — Solar Filament Segmentation Challenge 2026

## 1. Project & Objective
- **Task**: Instance segmentation of solar filaments from ground-based GONG H-Alpha images.
- **Goal**: Generate precise masks for individual filaments, capturing fine barbs while avoiding false positives and fragmentation.
- **Deadline**: Nov 15, 2026. Target PQ: > 0.35.

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
- **Multi-annotator quirk (CRITICAL)**: JSON has 1,154 image entries for 707 files; 296 files were annotated 2–3× independently (each set = separate entry). Never merge sets or treat entries as images — use `src/annotations.py` (one set per file, policy `first`; ~7.4 filaments/file).
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
