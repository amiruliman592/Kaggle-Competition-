# 🌞 Solar Filament Segmentation Challenge 2026 — Detailed Team Plan

---

## 1. Competition Overview

| Item | Detail |
|---|---|
| **Task** | Instance segmentation of solar filaments from GONG H-Alpha images |
| **Primary Metric** | **Panoptic Quality (PQ)** (leaderboard) |
| **Secondary Metrics** | Mean Dice Score, IoU distribution, 1→many / many→1 relations |
| **Scoring Rubric** | 70% Quantitative (PQ) + 30% Qualitative (report + code quality) |
| **Deadline** | Nov 15, 2026 (report + code) |
| **Prize Pool** | Up to $3,000 (top 3 teams) |
| **Target PQ** | > 0.35 per organizer guidance |

---

## 2. Dataset Summary

| Item | Value |
|---|---|
| **Train images** | 707 JPEG images |
| **Test images** | 180 JPEG images (no overlap with train) |
| **Image size** | 2048 × 2048 pixels (fixed) |
| **Annotation file** | `MAGFiLO_1.0_Annotations_kaggle2026_train.json` (~48 MB COCO-format) |
| **Total annotations** | 8,199 filament instances |
| **Avg filaments/image** | ~7.1 (min: 1, max: 26) |
| **Filament area** | min: 9 px², max: 37,739 px², mean: ~2,120 px² |
| **Segmentation format** | Single-polygon per instance (COCO polygon, all `iscrowd=0`) |
| **Categories** | Left (1), Right (2), Unidentifiable (3), Ambiguous (4) |
| **Extra metadata** | Spine annotations available for all 8,199 instances |
| **Output format** | CSV with `filament_id` + `segmentation_rle` (pycocotools RLE) |

> [!NOTE]
> Test images have **no annotation file** — you only get the raw JPEG images.
> The segmentation format in annotations is polygon (x,y points), but submission expects **RLE** encoding.

---

## 3. Key Challenges

1. **Fine-scale structures**: Barbs (thin thread-like extensions) are hard to capture
2. **Background noise**: Ground-based GONG observatory → atmospheric artifacts, noise, lens artifacts
3. **Structural continuity**: Models tend to fragment long filaments or merge nearby ones
4. **Very small objects**: Areas as small as 9 px² — standard anchor/pooling sizes may miss them
5. **Large image resolution**: 2048×2048 — requires tiling or memory-efficient inference strategies
6. **Instance vs semantic**: Must separate *individual* filaments, not just binary masks

---

## 4. Evaluation Deep-Dive

### Panoptic Quality (PQ)

```
PQ = SQ × RQ

where:
  SQ (Segmentation Quality) = avg IoU over true positives
  RQ (Recognition Quality) = TP / (TP + 0.5*FP + 0.5*FN)
```

A prediction matches a ground truth if **IoU > 0.5** (standard PQ threshold).

> [!IMPORTANT]
> **Fragmentation hurts RQ badly.** If one filament is predicted as 3 fragments → 2 extra FPs. 
> **Over-merging also hurts.** One prediction covering 2 filaments → 1 FN.
> This means we must prioritize **completeness** of individual filaments over pixel-perfect edges.

---

## 5. Submission Format

```
filament_id              | segmentation_rle
-------------------------|-------------------
20150125172714Mh_1       | f8uSDds...VQNC
20150125172714Mh_2       | KHT%$HD...9>km
...
```

- Image id = filename without extension (e.g. `20150125172714Mh`)
- Each row = one predicted filament instance
- RLE = **pycocotools RLE Counts** (not the size, just the counts string)
- Size is fixed 2048×2048 for all images

---

## 6. Recommended Pipeline (Ranked by Expected Performance)

### 🏆 Option A — Deep Learning: Mask2Former / SAM2 Fine-tuned (Recommended)

This is the highest-performing expected approach.

```
Input Image (2048×2048 JPEG)
        ↓
[Preprocessing]
  - Circular solar disk crop / mask
  - Normalize (per-image or dataset-wide)
  - Optional: CLAHE contrast enhancement
        ↓
[Tiling Strategy]  ← 2048 is too large for most GPU VRAM
  - Split into overlapping tiles (e.g. 512×512 or 1024×1024, stride=256/512)
  - OR resize to 1024×1024 (risk: lose fine structures)
        ↓
[Instance Segmentation Model]
  Option A1: Mask2Former (best COCO instance seg model)
  Option A2: SAM2 (Segment Anything Model 2, prompt-free with ViT backbone)
  Option A3: YOLOv8/v11-seg (fast, good baseline)
  Option A4: Detectron2 / PointRend
        ↓
[Tile Merging] ← stitch tile predictions with NMS
        ↓
[Post-Processing]
  - Remove very small predictions (< threshold area)
  - Morphological closing to fill gaps in filaments
  - Connected component analysis to merge near-fragments
        ↓
[Output: Instance masks]
        ↓
[RLE Encoding (pycocotools)]
        ↓
[CSV Submission]
```

### 🥈 Option B — Traditional CV + ML Hybrid (Good Baseline, No GPU needed)

```
Input Image
        ↓
[Preprocessing]
  - CLAHE enhancement
  - Gaussian blur + threshold (Otsu / local adaptive)
  - Morphological operations (open/close)
        ↓
[Region Proposals]
  - Connected components on thresholded image
  - OR Watershed segmentation
        ↓
[Feature Extraction per Region]
  - Area, aspect ratio, elongation, solidity, intensity stats
  - HOG, LBP descriptors
        ↓
[Classifier]
  - Random Forest / XGBoost: filament vs non-filament
        ↓
[Instance separation]
  - Spine-guided splitting (if spine prediction is added)
  - Distance transform watershed
        ↓
[Post-processing + RLE encoding]
```

### 🥉 Option C — Zero-shot SAM2 (Quick Baseline, No Training)

Use SAM2 in automatic mask generation mode without any fine-tuning. Good for establishing a fast baseline.

---

## 7. Detailed Step-by-Step Plan

### Phase 1: Setup & EDA (Days 1–3)

**Tasks:**
- [ ] Install environment: `torch`, `detectron2`/`mmdetection`, `pycocotools`, `opencv-python`, `albumentations`
- [ ] Load and visualize annotation JSON (COCO format)
- [ ] Plot sample images with their polygon masks overlaid
- [ ] Analyze filament size distribution (area histogram)
- [ ] Analyze filament count per image distribution
- [ ] Visualize spine annotations overlaid on images
- [ ] Check for bad/noisy annotations (very small area < 50 px²)
- [ ] Look at edge cases: max-filament images (26 filaments), smallest instances
- [ ] Understand the circular solar disk boundary in images

**Key Questions to Answer:**
- What fraction of images have the full disk visible vs partial?
- Do dark limb effects affect filament detection near disk edges?
- Are there duplicate/near-duplicate images (same day, different observatories)?
- How often do filaments span tile boundaries?

---

### Phase 2: Data Preprocessing Pipeline (Days 3–6)

**Tasks:**
- [ ] **Solar disk detection**: Hough circle transform to find disk center + radius
- [ ] **Disk masking**: Zero out pixels outside the solar disk
- [ ] **Normalize intensity**: Per-image normalization (subtract mean, divide std) OR global dataset normalization
- [ ] **CLAHE enhancement**: Contrast Limited Adaptive Histogram Equalization to boost local contrast
- [ ] **Convert polygon → binary mask**: Use pycocotools `annToMask` for training
- [ ] **Tiling strategy**: Design 512×512 or 1024×1024 tile extraction with overlap
- [ ] **Data augmentation plan**: flips (H/V), rotation, brightness/contrast jitter, elastic distortion
- [ ] **Train/validation split**: Stratified by year or by number of filaments per image

**Recommended Split:**
- 80% train (~566 images), 20% val (~141 images)
- Keep temporally contiguous blocks in val to simulate test distribution

---

### Phase 3: Baseline Model (Days 6–12)

**Recommended: YOLOv8-seg or Mask R-CNN**

**Tasks:**
- [ ] Set up Mask R-CNN (detectron2) OR YOLOv8-seg training configuration
- [ ] Convert COCO polygon annotations to training format
- [ ] Train on resized 1024×1024 images (fast iteration)
- [ ] Evaluate on val set: compute PQ, Dice, IoU
- [ ] Implement self-evaluation notebook metrics (from Kaggle notebook)
- [ ] Submit baseline prediction CSV to Kaggle

**Config choices for Mask R-CNN baseline:**
```python
# Key hyperparameters to tune:
ANCHOR_SIZES = [[8], [16], [32], [64], [128]]  # Small anchors for tiny filaments
MIN_SIZE_TRAIN = 1024
MAX_SIZE_TRAIN = 1024
ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
MODEL.RESNETS.DEPTH = 50  # Start with R50, upgrade to R101
```

---

### Phase 4: Advanced Model — Mask2Former (Days 12–25)

**Why Mask2Former?**
- State-of-the-art on COCO panoptic/instance segmentation
- Transformer-based decoder handles object queries well
- Better at small, thin, elongated objects vs anchor-based methods

**Tasks:**
- [ ] Set up Mask2Former (via `mmdetection` or HuggingFace `transformers`)
- [ ] Use pretrained backbone: Swin-T or Swin-S (pretrained on ImageNet-22K)
- [ ] Fine-tune on solar filament data (start with LR=1e-4, backbone LR=1e-5)
- [ ] Implement tiling: run model on tiles, stitch results
- [ ] Stitch strategy: for predictions at tile boundaries, apply NMS/WBF (Weighted Box Fusion)
- [ ] Validate PQ score on val set

**Training config sketch:**
```python
# Mask2Former key config
model:
  backbone: swin_tiny_patch4_window7_224
  num_queries: 100  # max filaments detectable per tile
  hidden_dim: 256
  nheads: 8
  dim_feedforward: 2048
  dec_layers: 9  # transformer decoder layers
  
optimizer: AdamW, lr=1e-4, weight_decay=0.05
scheduler: MultiStepLR [step 8, 11], gamma=0.1
batch_size: 4 (with gradient accumulation if VRAM limited)
epochs: 50
```

---

### Phase 5: SAM2 Exploration (Days 15–20, parallel)

**Tasks:**
- [ ] Run SAM2 in automatic mask generation mode (zero-shot) on val set
- [ ] Measure PQ score → this is your "free" baseline
- [ ] Attempt few-shot prompting: use centroid of GT as point prompt
- [ ] If promising: fine-tune SAM2 on solar filament data (SAM2 supports fine-tuning)

---

### Phase 6: Post-Processing & Ensemble (Days 25–35)

This phase often provides the biggest PQ jumps:

**Post-processing techniques:**
- [ ] **Morphological closing**: Fill holes/gaps in predicted masks (helps with fragmentation)
- [ ] **Spine-guided merging**: If predicted masks are within distance D of each other and share spine orientation, merge them
- [ ] **Area filtering**: Remove predictions with area < threshold (tune on val)
- [ ] **Confidence thresholding**: Tune score threshold per model
- [ ] **Test-Time Augmentation (TTA)**: Run inference with H-flip, V-flip, multi-scale; merge masks with NMS
- [ ] **Ensemble**: Combine Mask2Former + YOLOv8 predictions with NMS/WBF

**NMS / merging strategy for tiled predictions:**
```python
# After collecting all tile predictions:
1. Convert all masks to global 2048x2048 coordinates
2. Compute pairwise IoU between all predicted masks
3. Run NMS with IoU threshold=0.3 (lower than typical 0.5 since filaments can be adjacent)
4. Keep highest-confidence mask per cluster
```

---

### Phase 7: Spine-Guided Approach (Optional, High Risk / High Reward)

The dataset includes spine annotations. We could:

- [ ] Train a separate **spine prediction network** (e.g. LSQ-Net, DEXTR) 
- [ ] Use predicted spines as **guidance for mask generation** (spine → bounding proposal → mask)
- [ ] This directly targets the "structural continuity" challenge
- [ ] Use spine as a prompt for SAM2 (spine points → SAM2 mask prediction)

---

### Phase 8: Final Submission & Report (Days 35–45)

**Tasks:**
- [ ] Final model selection based on val PQ
- [ ] Generate predictions on all 180 test images
- [ ] Validate submission CSV format (check image IDs, RLE format)
- [ ] Submit to Kaggle and verify leaderboard score
- [ ] Write 4-page technical report (Overleaf template provided)
- [ ] Clean and document code repository
- [ ] Add `requirements.txt` with pinned versions
- [ ] Create end-to-end Jupyter notebook for reproducing results
- [ ] Make GitHub repository public

---

## 8. Metric Implementation

```python
# Self-evaluation (mirrors Kaggle scoring)
from torchmetrics.segmentation import DiceScore
from torchmetrics.detection import PanopticQuality

# PQ requires matching predictions to GT
# IoU threshold = 0.5 for TP determination
# TP: IoU(pred_i, gt_j) > 0.5
# FP: predicted instances with no matching GT
# FN: GT instances with no matching prediction

# SQ = mean IoU over TPs
# RQ = TP / (TP + 0.5*FP + 0.5*FN)
# PQ = SQ * RQ
```

---

## 9. RLE Encoding (Submission Format)

```python
import numpy as np
from pycocotools import mask as coco_mask

def mask_to_rle(binary_mask: np.ndarray) -> str:
    """Convert binary mask (H×W) to COCO RLE counts string."""
    # binary_mask: shape (2048, 2048), dtype uint8
    rle = coco_mask.encode(np.asfortranarray(binary_mask))
    return rle['counts'].decode('utf-8')

def build_submission(predictions: dict) -> list:
    """
    predictions: {image_id: [mask1, mask2, ...]}
    Returns list of dicts for CSV.
    """
    rows = []
    for img_id, masks in predictions.items():
        for i, mask in enumerate(masks, start=1):
            rle_str = mask_to_rle(mask)
            rows.append({
                'filament_id': f'{img_id}_{i}',
                'segmentation_rle': rle_str
            })
    return rows
```

---

## 10. Technology Stack

| Component | Recommended Tool |
|---|---|
| **Primary framework** | PyTorch + detectron2 OR mmdetection |
| **Instance seg model** | Mask2Former (primary), YOLOv8-seg (baseline) |
| **Zero-shot baseline** | SAM2 (Meta) |
| **Data loading** | pycocotools, PIL/OpenCV |
| **Augmentation** | albumentations |
| **Metrics** | torchmetrics |
| **RLE encoding** | pycocotools |
| **Post-processing** | OpenCV, scipy |
| **Experiment tracking** | Weights & Biases (wandb) |
| **Report** | Overleaf (LaTeX) |

---

## 11. Risk Assessment & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| GPU memory OOM at 2048×2048 | High | High | Use tiling strategy |
| Fragmentation of long filaments | High | High | Morphological post-proc, spine guidance |
| Over-merging nearby filaments | Medium | High | Tune NMS IoU threshold carefully |
| Very small filaments missed | Medium | Medium | Small anchor sizes, high-res tiles |
| Domain shift train→test | Low | High | Diverse augmentation, no test-time labels |
| Tile boundary artifacts | Medium | Medium | Overlapping tiles + NMS stitching |

---

## 12. Timeline Summary

```
Week 1  (Oct 13–19): EDA + environment setup + data preprocessing + disk masking
Week 2  (Oct 20–26): YOLOv8-seg baseline + val evaluation + first Kaggle submission
Week 3  (Oct 27–Nov 2): Mask2Former training + tiling strategy
Week 4  (Nov 3–9):   Post-processing + TTA + ensemble + SAM2 exploration
Week 5  (Nov 10–14): Final model selection + submission polishing + report writing
Nov 15: SUBMISSION DEADLINE
```

---

## 13. Team Role Suggestions

| Role | Responsibilities |
|---|---|
| **Data Engineer** | EDA, preprocessing pipeline, disk masking, tiling, data loading |
| **ML Engineer 1** | Mask2Former / Detectron2 training, hyperparameter tuning |
| **ML Engineer 2** | YOLOv8-seg baseline, SAM2 zero-shot, post-processing |
| **Evaluation Lead** | PQ/Dice metric implementation, self-evaluation, Kaggle submissions |
| **Report Author** | Technical report writing (LaTeX), GitHub repo documentation |

> [!TIP]
> Roles can overlap. The most impactful work is: (1) correct tiling+stitching, (2) post-processing to fix fragmentation, (3) Mask2Former fine-tuning. Prioritize these.

---

## 14. Quick Wins (Easy PQ Improvements)

1. **Disk masking** — zero out non-solar pixels → eliminates false positives near corners
2. **CLAHE** — significantly improves contrast for threshold-based and neural approaches
3. **Area filtering** — remove tiny predicted blobs (< 50 px²) → reduces FP count
4. **TTA with horizontal flip** — free ~1-2% PQ improvement for neural models
5. **Morphological closing** (kernel 5×5) after prediction → reduces FN from fragmentation

---

## 15. References

- Competition: https://www.kaggle.com/competitions/filament-segmentation-2026
- MAGFiLO dataset paper: doi:10.1038/s41597-024-03876-y
- Panoptic Quality paper: doi:10.1109/CVPR.2019.00963
- Self-evaluation notebook: https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook
- pycocotools: https://github.com/cocodataset/cocoapi
- Mask2Former: https://github.com/facebookresearch/Mask2Former
- SAM2: https://github.com/facebookresearch/segment-anything-2
