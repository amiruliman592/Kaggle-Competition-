# 🌞 Solar Filament Instance Segmentation (Kaggle 2026)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Detectron2](https://img.shields.io/badge/Detectron2-v0.6-brightgreen.svg)](https://github.com/facebookresearch/detectron2)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end instance segmentation pipeline designed for the **Solar Filament Segmentation Challenge 2026**. This repository implements a full Detectron2 (Mask R-CNN ResNet-50 FPN) baseline with customized anchor scales, solar disk masking, competition-exact RLE encoding, and local Panoptic Quality (PQ) offline evaluation.

---

## 📌 Table of Contents

- [Competition Summary](#-competition-summary)
- [Project Architecture](#-project-architecture)
- [Quick Start Guide](#-quick-start-guide)
  - [Option A: Google Colab Pro / Kaggle GPU (Recommended)](#option-a-google-colab-pro--kaggle-gpu-recommended)
  - [Option B: Local Machine (EDA & Offline Evaluation)](#option-b-local-machine-eda--offline-evaluation)
  - [Option C: WSL2 / Linux GPU](#option-c-wsl2--linux-gpu)
- [Pipeline Usage](#-pipeline-usage)
- [Module Overview](#-module-overview)
- [Evaluation & Metric](#-evaluation--metric)
- [Submission Format](#-submission-format)
- [Next Steps & Team Roadmap](#-next-steps--team-roadmap)
- [Contributing Guidelines](#-contributing-guidelines)

---

## 🔬 Competition Summary

| Item | Details |
|---|---|
| **Goal** | Segment individual solar filaments from ground-based GONG $H\alpha$ images |
| **Primary Metric** | **Panoptic Quality (PQ = SQ × RQ)** |
| **Image Resolution** | Fixed $2048 \times 2048$ pixels (JPEG) |
| **Dataset Size** | 707 training images (8,199 filament instances), 180 test images |
| **Key Challenges** | Fine barb threads, background atmospheric noise, filament fragmentation, extreme size variations (9 px² to 37,000+ px²) |

---

## 🗂️ Project Architecture

```text
Solar_Filament_Kaggle/
├── .gitignore                          # Ignores __pycache__, checkpoints, output/
├── requirements.txt                    # Project dependencies
├── run_pipeline.py                     # Master CLI runner (EDA, Train, Inference)
├── Solar_Filament_Segmentation_Plan.md # In-depth strategic team plan & roadmap
├── filament-segmentation-2026/         # Dataset directory (Kaggle format)
│   └── MAGFiLO_1.0_Kaggle_2026/
│       ├── train/
│       │   ├── train_images/
│       │   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json
│       └── test/
│           └── test_images/
├── src/                                # Core codebase
│   ├── __init__.py
│   ├── config.py                       # Centralized hyperparameters & paths
│   ├── dataset.py                      # COCO JSON loader, solar disk mask, train/val split
│   ├── eda.py                          # Exploratory data analysis & visualizations
│   ├── train.py                        # Detectron2 Mask R-CNN training & evaluator
│   ├── evaluate.py                     # Offline PQ, SQ, RQ, Dice, and IoU evaluator
│   ├── inference.py                    # Test inference & pycocotools RLE generator
│   └── visualize.py                    # Qualitative prediction overlay visualizer
└── output/                             # Auto-generated checkpoints, EDA plots, submission.csv
```

---

## ⚡ Quick Start Guide

### Option A: Google Colab Pro / Kaggle GPU (Recommended)

Since Detectron2 requires CUDA C++ compilation and model training benefits greatly from an A100 or T4/P100 GPU:

1. Clone or upload this repository to Google Drive / Colab or Kaggle:
   ```bash
   git clone <REPO_URL>
   cd Solar_Filament_Kaggle
   ```

2. Run the full automated pipeline with the `--colab` flag:
   ```bash
   python run_pipeline.py --colab --step all
   ```
   This will:
   - Install required dependencies and Detectron2.
   - Run Exploratory Data Analysis (EDA) and save plots to `output/eda/`.
   - Train the Mask R-CNN model on the 80/20 train/val split.
   - Run inference across all 180 test images.
   - Export `output/submission.csv` ready for Kaggle submission.

---

### Option B: Local Machine (EDA & Offline Evaluation)

You can run EDA, inspect datasets, and calculate metrics locally without needing GPU/Detectron2:

1. Create and activate a virtual environment (Python 3.10 recommended):
   ```powershell
   conda create -n solar_filament python=3.10 -y
   conda activate solar_filament
   ```

2. Install core visualization and data processing packages:
   ```powershell
   pip install -r requirements.txt
   ```

3. Run EDA:
   ```powershell
   python src/eda.py
   ```

---

### Option C: WSL2 / Linux GPU

If training locally on Windows with an NVIDIA GPU, use WSL2 (Ubuntu):

```bash
# 1. Install PyTorch with matching CUDA (e.g., cu121)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install Detectron2 from official GitHub
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 3. Install other dependencies
pip install -r requirements.txt

# 4. Run pipeline
python run_pipeline.py --step all
```

---

## 🚀 Pipeline Usage

[run_pipeline.py](file:///c:/Users/User/Downloads/Solar_Filament_Kaggle/run_pipeline.py) provides flexible stage-by-stage execution via `--step`:

```bash
# Setup dependencies (Colab)
python run_pipeline.py --step setup

# Step 1: Run Exploratory Data Analysis (generates figures in output/eda/)
python run_pipeline.py --step eda

# Step 2: Train Mask R-CNN baseline
python run_pipeline.py --step train

# Step 3: Run inference on test images & generate output/submission.csv
python run_pipeline.py --step infer

# Run all steps sequentially
python run_pipeline.py --step all
```

---

## 🧩 Module Overview

### 1. Central Configuration ([src/config.py](file:///c:/Users/User/Downloads/Solar_Filament_Kaggle/src/config.py))
Stores all parameters in a clean `@dataclass Config` instance (`CFG`):
- **Image Resolution**: Native $(2048, 2048)$, resized to $(1024, 1024)$ during training for memory efficiency.
- **Model Backbone**: `mask_rcnn_R_50_FPN_3x`.
- **Anchor Tuning**: Fine-scale anchors `[[8], [16], [32], [64], [128]]` specifically configured for thin and small filament instances (down to 9 px²).
- **Hyperparameters**: Base LR $0.0025$, batch size $2$, max iterations $3000$, evaluation period $500$.

### 2. Dataset Preparation ([src/dataset.py](file:///c:/Users/User/Downloads/Solar_Filament_Kaggle/src/dataset.py))
- Reads the COCO-formatted `MAGFiLO_1.0_Annotations_kaggle2026_train.json`.
- Implements `get_train_val_split()` with reproducible random seeding.
- Provides `create_solar_disk_mask()` using Hough Circle / Otsu thresholding to mask out space noise outside the solar perimeter.
- Registers standard Detectron2 dataset catalogs (`filament_train`, `filament_val`).

### 3. Exploratory Data Analysis ([src/eda.py](file:///c:/Users/User/Downloads/Solar_Filament_Kaggle/src/eda.py))
Generates:
- `output/eda/sample_ground_truth.png`: Bounding boxes and instance contours overlaid on original solar disks.
- `output/eda/area_distribution.png`: Log-scale filament area distributions.
- `output/eda/filaments_per_image.png`: Instance density per image (averaging ~7.1 filaments/image).
- `output/eda/category_distribution.png`: Left, Right, Unidentifiable, and Ambiguous breakdowns.

### 4. Training Engine ([src/train.py](file:///c:/Users/User/Downloads/Solar_Filament_Kaggle/src/train.py))
- Custom `FilamentTrainer` subclassing Detectron2's `DefaultTrainer`.
- Custom `DatasetMapper` integrating random horizontal/vertical flips and resize transforms.
- Periodic COCO evaluation on the validation set with automated checkpoint saving in `output/`.

### 5. Offline Evaluation ([src/evaluate.py](file:///c:/Users/User/Downloads/Solar_Filament_Kaggle/src/evaluate.py))
Enables local metric tracking without burning Kaggle daily submission limits:
- Implements exact competition **Panoptic Quality (PQ)**:
  $$\text{PQ} = \text{SQ} \times \text{RQ}$$
  $$\text{SQ} = \frac{\sum_{(p, g) \in \text{TP}} \text{IoU}(p, g)}{|\text{TP}|}, \quad \text{RQ} = \frac{|\text{TP}|}{|\text{TP}| + \frac{1}{2}|\text{FP}| + \frac{1}{2}|\text{FN}|}$$
- Computes Mean Dice and IoU statistics across all validation images.

### 6. Inference & Submission ([src/inference.py](file:///c:/Users/User/Downloads/Solar_Filament_Kaggle/src/inference.py))
- Loads the trained model weights.
- Processes all 180 test images.
- Scales masks back to native $2048 \times 2048$ resolution.
- Applies `pycocotools.mask.encode(np.asfortranarray(mask))['counts']`.
- Exports `output/submission.csv` strictly compliant with competition formatting.

---

## 📊 Submission Format

The submission file matches the official Kaggle requirement:

| `filament_id` | `segmentation_rle` |
|---|---|
| `20150125172714Mh_1` | `f8uSDds...VQNC` |
| `20150125172714Mh_2` | `KHT%$HD...9>km` |

- `filament_id`: `{image_name_without_extension}_{instance_index}` (1-indexed).
- `segmentation_rle`: pycocotools RLE counts string.
- Empty predictions for an image are omitted; images with multiple filaments have multiple rows.

---

## 🗺️ Next Steps & Team Roadmap

Collaborators looking to push leaderboard score can focus on the prioritized tasks below:

1. **Post-Processing Optimization (High Impact / Easy)**:
   - **Fragmentation Fix**: Filaments often break into multiple pieces, which severely hurts Recognition Quality (RQ). Implement morphological closing (`cv2.morphologyEx`) or proximity clustering to merge fragments.
   - **Small Blob Filtering**: Filter out tiny artifact predictions $< 30\text{ px}^2$ to eliminate false positives.
   - **Disk Mask Filtering**: Discard any prediction centered outside the solar disk radius.
2. **2048×2048 Tiling & Stitching (Medium Impact)**:
   - Instead of resizing $2048 \to 1024$ (which loses thin filament barbs), train on overlapping $1024 \times 1024$ patches with stride $512$ and stitch with Non-Maximum Suppression (NMS).
3. **Model Upgrades**:
   - Experiment with **Mask2Former** (SOTA on instance segmentation).
   - Test **YOLOv8x-seg / YOLOv11-seg** for rapid iteration.
   - Evaluate zero-shot prompt-based segmentation with **SAM2**.
4. **Test-Time Augmentation (TTA)**:
   - Horizontal and vertical flip averaging during inference for an easy $+0.01\sim 0.02$ boost in PQ.

---

## 👥 Contributing Guidelines

1. **Branching**:
   - Create a feature branch: `git checkout -b feature/<your-name>-<feature-description>` (e.g. `feature/ali-morphological-postproc`).
2. **Keep Configs Centralized**:
   - Avoid hardcoding parameters in individual scripts; add new hyperparameters to [src/config.py](file:///c:/Users/User/Downloads/Solar_Filament_Kaggle/src/config.py).
3. **Commit Cleanly**:
   - Do **NOT** commit trained weights (`*.pth`), large outputs, or local virtual environments (handled by `.gitignore`).
4. **Evaluate Before Submitting**:
   - Always run `python src/evaluate.py` to confirm your local validation PQ improves before generating a new `submission.csv`.
