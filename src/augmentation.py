"""
augmentation.py
---------------
Albumentations-based augmentation pipelines for Brain MRI segmentation.

Usage:
    from augmentation import get_train_transform, get_val_transform

    train_tfm = get_train_transform(img_size=256)
    val_tfm   = get_val_transform(img_size=256)
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2   # optional — we handle tensors in Dataset


# ======================================================================
#  Training transforms  (heavy augmentation)
# ======================================================================

def get_train_transform(img_size: int = 256) -> A.Compose:
    return A.Compose([

        # ── Spatial ───────────────────────────────────────────────────
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        A.Affine(
            translate_percent=0.1,
            scale=(0.85, 1.15),
            rotate=(-45, 45),
            p=0.7,
        ),

        A.ElasticTransform(
            alpha=120,
            sigma=6,
            p=0.3,
        ),

        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            p=0.2,
        ),

        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            p=0.3,
        ),

        # ── Intensity ─────────────────────────────────────────────────
        A.GaussNoise(p=0.3),
        A.RandomBrightnessContrast(p=0.5),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

        # ── Final resize ──────────────────────────────────────────────
        A.Resize(img_size, img_size),

    ], additional_targets={"mask": "mask"})


# ======================================================================
#  Validation / test transforms  (no augmentation, just resize)
# ======================================================================

def get_val_transform(img_size: int = 256) -> A.Compose:
    """
    Minimal pipeline for validation and test sets.
    Only resizes — no stochastic augmentation.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
    ], additional_targets={"mask": "mask"})


# ======================================================================
#  TTA (Test-Time Augmentation) transforms  — used in Week 5
# ======================================================================

def get_tta_transforms(img_size: int = 256) -> list[A.Compose]:
    """
    Returns a list of deterministic transforms for TTA inference.
    Predictions are averaged across all augmented versions.
    """
    base = A.Resize(img_size, img_size)
    return [
        A.Compose([base]),
        A.Compose([A.HorizontalFlip(p=1.0), base]),
        A.Compose([A.VerticalFlip(p=1.0),   base]),
        A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0), base]),
        A.Compose([A.Rotate(limit=(90, 90), p=1.0), base]),
        A.Compose([A.Rotate(limit=(270, 270), p=1.0), base]),
    ]


# ======================================================================
#  Quick visual check
# ======================================================================

if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    # Change this to any image-mask pair in your dataset
    SAMPLE_IMG  = "../data/raw/kaggle_3m/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_1.tif"
    SAMPLE_MASK = "../data/raw/kaggle_3m/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_1_mask.tif"

    img  = cv2.cvtColor(cv2.imread(SAMPLE_IMG),  cv2.COLOR_BGR2RGB)
    mask = cv2.imread(SAMPLE_MASK, cv2.IMREAD_GRAYSCALE)

    train_tfm = get_train_transform(256)

    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    for col in range(6):
        out  = train_tfm(image=img, mask=mask)
        axes[0, col].imshow(out["image"]);            axes[0, col].set_title(f"Aug {col+1}")
        axes[1, col].imshow(out["mask"], cmap="gray"); axes[1, col].set_title("Mask")

    for ax in axes.flat:
        ax.axis("off")

    plt.suptitle("Augmentation Preview (6 random samples)", fontsize=13)
    plt.tight_layout()
    plt.savefig("augmentation_preview.png", dpi=120)
    print("Saved augmentation_preview.png")