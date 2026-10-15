"""
run_pipeline.py — End-to-end pipeline: EDA → Train → Inference → Submission.

This is the main entry point designed to be run on Google Colab Pro.
Copy this file to Colab and run it cell-by-cell, or run the whole thing:

  python run_pipeline.py

Steps:
  1. Install dependencies (if on Colab)
  2. Run EDA
  3. Train Mask R-CNN
  4. Run inference on test images
  5. Generate submission CSV
"""

import os
import sys
import subprocess

# ═══════════════════════════════════════════════════════════════════════
# STEP 0: Setup (run this on Colab)
# ═══════════════════════════════════════════════════════════════════════

def setup_colab():
    """Install dependencies on Google Colab."""
    print("=" * 60)
    print("  STEP 0: Installing dependencies")
    print("=" * 60)
    
    # Install core packages
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "pycocotools", "albumentations", "torchmetrics", "seaborn", "tqdm",
        "scikit-image",
    ])
    
    # Install Detectron2 (must match CUDA version)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "git+https://github.com/facebookresearch/detectron2.git",
    ])
    
    print("[Setup] All dependencies installed!")


def check_environment():
    """Verify the environment is ready."""
    import torch
    print(f"  Python:     {sys.version.split()[0]}")
    print(f"  PyTorch:    {torch.__version__}")
    print(f"  CUDA:       {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})")
    
    try:
        import detectron2
        print(f"  Detectron2: {detectron2.__version__}")
    except ImportError:
        print("  Detectron2: NOT INSTALLED — run setup_colab() first!")
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: EDA
# ═══════════════════════════════════════════════════════════════════════

def run_eda():
    """Run exploratory data analysis."""
    print("\n" + "=" * 60)
    print("  STEP 1: Exploratory Data Analysis")
    print("=" * 60)
    
    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from eda import main as eda_main
    eda_main()


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Train
# ═══════════════════════════════════════════════════════════════════════

def run_training():
    """Train Mask R-CNN model."""
    print("\n" + "=" * 60)
    print("  STEP 2: Training Mask R-CNN")
    print("=" * 60)
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from train import train
    cfg = train()
    return cfg


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Inference & Submission
# ═══════════════════════════════════════════════════════════════════════

def run_inference_and_submit():
    """Run inference on test images and create submission CSV."""
    print("\n" + "=" * 60)
    print("  STEP 3: Inference & Submission")
    print("=" * 60)
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from inference import main as inference_main
    inference_main()


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Solar Filament Segmentation Pipeline")
    parser.add_argument("--step", type=str, default="all",
                        choices=["setup", "eda", "train", "infer", "all"],
                        help="Which step to run (default: all)")
    parser.add_argument("--colab", action="store_true",
                        help="Running on Google Colab (installs dependencies)")
    args = parser.parse_args()
    
    if args.colab or args.step == "setup":
        setup_colab()
    
    print("\n[Pipeline] Environment check:")
    ready = check_environment()
    
    if args.step in ("eda", "all"):
        run_eda()
    
    if args.step in ("train", "all"):
        if not ready:
            print("[Pipeline] Cannot train without Detectron2. Run --step setup first.")
            sys.exit(1)
        run_training()
    
    if args.step in ("infer", "all"):
        if not ready:
            print("[Pipeline] Cannot infer without Detectron2. Run --step setup first.")
            sys.exit(1)
        run_inference_and_submit()
    
    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("  Submission file: output/submission.csv")
    print("=" * 60)
