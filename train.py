"""
🚦 Smart Traffic Management — YOLOv8 Model Training Script
==========================================================
Train a YOLOv8 model to detect vehicles from your dataset.

Usage:
    python train.py

Make sure the dataset is extracted to the 'dataset/' folder before running.
"""

from ultralytics import YOLO
import os
import shutil

# ──────────────────────────────────────────────────────────────
# CONFIGURATION — Adjust these settings as needed
# ──────────────────────────────────────────────────────────────

DATASET_YAML = "data.yaml"             # Path to your dataset config
MODEL_BASE = "yolov8n.pt"              # Base model (nano = fastest, good for starting)
EPOCHS = 50                            # Number of training epochs
IMG_SIZE = 640                         # Image size for training
BATCH_SIZE = 8                         # Batch size (reduce to 4 if you get memory errors)
PROJECT_NAME = "runs/train"            # Output directory for training results
EXPERIMENT_NAME = "traffic_model"      # Name for this training run

# ──────────────────────────────────────────────────────────────
# PRE-FLIGHT CHECKS
# ──────────────────────────────────────────────────────────────

def check_dataset():
    """Verify the dataset exists and is properly structured."""
    if not os.path.exists(DATASET_YAML):
        print("❌ ERROR: Dataset not found!")
        print(f"   Expected: {os.path.abspath(DATASET_YAML)}")
        print()
        print("   Please make sure you:")
        print("   1. Downloaded the dataset from Roboflow")
        print("   2. Extracted the ZIP into the 'dataset/' folder")
        print("   3. The folder structure looks like:")
        print("      dataset/")
        print("      ├── train/")
        print("      │   ├── images/")
        print("      │   └── labels/")
        print("      ├── valid/")
        print("      │   ├── images/")
        print("      │   └── labels/")
        print("      └── data.yaml")
        return False

    # Read and display dataset info
    with open(DATASET_YAML, "r") as f:
        content = f.read()
    print("✅ Dataset found!")
    print("─" * 40)
    print("📄 data.yaml contents:")
    print(content)
    print("─" * 40)
    return True


def check_gpu():
    """Check if GPU is available for training."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detected: {gpu_name}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
            return True
        else:
            print("⚠️  No GPU detected — training will run on CPU (slower)")
            print("   This is fine, it will just take longer.")
            return False
    except ImportError:
        print("⚠️  PyTorch not found — will install with ultralytics")
        return False


# ──────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────

def train_model():
    """Train the YOLOv8 model."""
    print()
    print("=" * 50)
    print("🚀 STARTING MODEL TRAINING")
    print("=" * 50)
    print(f"   Base Model  : {MODEL_BASE}")
    print(f"   Dataset     : {DATASET_YAML}")
    print(f"   Epochs      : {EPOCHS}")
    print(f"   Image Size  : {IMG_SIZE}")
    print(f"   Batch Size  : {BATCH_SIZE}")
    print("=" * 50)
    print()

    # Load the base YOLOv8 model (downloads automatically if not present)
    model = YOLO(MODEL_BASE)

    # Train the model
    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        exist_ok=True,
        patience=10,          # Early stopping: stop if no improvement for 10 epochs
        save=True,            # Save checkpoints
        save_period=10,       # Save checkpoint every 10 epochs
        plots=True,           # Generate training plots
        verbose=True,
    )

    return results


def copy_best_model():
    """Copy the best model weights to the project root."""
    best_model_path = os.path.join(PROJECT_NAME, EXPERIMENT_NAME, "weights", "best.pt")

    if os.path.exists(best_model_path):
        shutil.copy2(best_model_path, "best.pt")
        print()
        print("=" * 50)
        print("✅ TRAINING COMPLETE!")
        print("=" * 50)
        print(f"   Best model saved to: best.pt")
        print(f"   Full results in   : {PROJECT_NAME}/{EXPERIMENT_NAME}/")
        print()
        print("   📊 Check these files for training metrics:")
        print(f"      - {PROJECT_NAME}/{EXPERIMENT_NAME}/results.png")
        print(f"      - {PROJECT_NAME}/{EXPERIMENT_NAME}/confusion_matrix.png")
        print()
        print("   🎯 Next step: Run 'streamlit run main.py' to use the model!")
        print("=" * 50)
    else:
        print("⚠️  Could not find best.pt — check the training output above for errors.")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("🚦 Smart Traffic Management — Model Trainer")
    print("=" * 50)
    print()

    # Step 1: Check dataset
    print("📂 Step 1: Checking dataset...")
    if not check_dataset():
        exit(1)

    # Step 2: Check GPU
    print()
    print("🖥️  Step 2: Checking hardware...")
    check_gpu()

    # Step 3: Train
    print()
    print("🏋️  Step 3: Training model...")
    train_model()

    # Step 4: Copy best model
    copy_best_model()
