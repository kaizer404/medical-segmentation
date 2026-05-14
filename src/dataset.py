"""
dataset.py
----------
Brain MRI Segmentation Dataset loader.
Kaggle dataset: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
Structure:
    data/raw/
        kaggle_3m/
            TCGA_<patient_id>/
                <patient_id>_<slice>.tif       ← MRI image
                <patient_id>_<slice>_mask.tif  ← binary mask
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader


class BrainMRIDataset(Dataset):
    """
    Loads (image, mask) pairs from the LGG MRI Segmentation dataset.

    Args:
        df         : DataFrame with columns ['image_path', 'mask_path']
        transform  : albumentations Compose transform (or None)
        img_size   : spatial size to resize to (H, W)
    """

    def __init__(self, df: pd.DataFrame, transform=None, img_size: int = 256):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_size = img_size

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.df)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Load image (RGB → keep 3 channels for pretrained encoder) ──
        image = cv2.imread(str(row["image_path"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))

        # ── Load mask (grayscale, binary 0/255) ──────────────────────
        mask = cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8)   # binarise → 0 or 1

        # ── Albumentations augmentation ───────────────────────────────
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"]

        # ── To tensor ─────────────────────────────────────────────────
        # image: H×W×3 → 3×H×W, float [0,1]
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        # mask:  H×W   → 1×H×W, float
        mask  = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask


# ======================================================================
#  Helper: build the file-pair DataFrame
# ======================================================================

def build_dataframe(data_root: str | Path) -> pd.DataFrame:
    """
    Recursively scans data_root for *_mask.tif files and pairs them
    with their corresponding image files.

    Returns a DataFrame with columns: ['image_path', 'mask_path',
                                        'patient_id', 'has_tumor']
    """
    data_root = Path(data_root)
    records = []

    for mask_path in sorted(data_root.rglob("*_mask.tif")):
        # image is the same filename without "_mask"
        img_name  = mask_path.name.replace("_mask.tif", ".tif")
        img_path  = mask_path.parent / img_name

        if not img_path.exists():
            continue

        # patient folder name == patient ID
        patient_id = mask_path.parent.name

        # Quick flag: non-zero mask → has tumour
        mask_arr   = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        has_tumor  = int(mask_arr.max() > 0) if mask_arr is not None else 0

        records.append({
            "image_path": img_path,
            "mask_path":  mask_path,
            "patient_id": patient_id,
            "has_tumor":  has_tumor,
        })

    df = pd.DataFrame(records)
    print(f"[dataset] Found {len(df)} image-mask pairs  "
          f"({df['has_tumor'].sum()} with tumour, "
          f"{(~df['has_tumor'].astype(bool)).sum()} without)")
    return df


# ======================================================================
#  Helper: stratified train/val/test split (patient-level)
# ======================================================================

def split_dataframe(
    df: pd.DataFrame,
    val_size:  float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Patient-level split so the same patient never appears in two splits.
    Stratified on whether the patient has ≥1 tumour slice.
    """
    # One row per patient, with tumour flag (1 if any slice has tumour)
    patient_df = (
        df.groupby("patient_id")["has_tumor"]
        .max()
        .reset_index()
    )

    train_patients, test_patients = train_test_split(
        patient_df["patient_id"],
        test_size=test_size,
        stratify=patient_df["has_tumor"],
        random_state=random_state,
    )
    remaining = patient_df[patient_df["patient_id"].isin(train_patients)]
    adjusted_val = val_size / (1 - test_size)

    train_patients, val_patients = train_test_split(
        remaining["patient_id"],
        test_size=adjusted_val,
        stratify=remaining["has_tumor"],
        random_state=random_state,
    )

    train_df = df[df["patient_id"].isin(train_patients)]
    val_df   = df[df["patient_id"].isin(val_patients)]
    test_df  = df[df["patient_id"].isin(test_patients)]

    print(f"[split] train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    return train_df, val_df, test_df


# ======================================================================
#  Helper: build DataLoaders
# ======================================================================

def get_dataloaders(
    data_root:       str | Path,
    train_transform = None,
    val_transform   = None,
    img_size:        int = 256,
    batch_size:      int = 16,
    num_workers:     int = 4,
) -> dict[str, DataLoader]:
    """
    One-liner to get {'train': ..., 'val': ..., 'test': ...} loaders.
    """
    df = build_dataframe(data_root)
    train_df, val_df, test_df = split_dataframe(df)

    datasets = {
        "train": BrainMRIDataset(train_df, transform=train_transform, img_size=img_size),
        "val":   BrainMRIDataset(val_df,   transform=val_transform,   img_size=img_size),
        "test":  BrainMRIDataset(test_df,  transform=val_transform,   img_size=img_size),
    }

    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
        for split, ds in datasets.items()
    }
    return loaders


# ======================================================================
#  Quick sanity-check
# ======================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # NEW
    DATA_ROOT = Path("../data/raw/kaggle_3m")       # ← adjust if needed

    df = build_dataframe(DATA_ROOT)
    print(df.head())

    # Show one positive sample
    pos = df[df["has_tumor"] == 1].iloc[0]
    img  = cv2.cvtColor(cv2.imread(str(pos["image_path"])), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(pos["mask_path"]), cv2.IMREAD_GRAYSCALE)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img);               axes[0].set_title("MRI")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Mask")
    overlay = img.copy()
    overlay[mask > 0] = [255, 0, 0]
    axes[2].imshow(overlay);           axes[2].set_title("Overlay")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    plt.savefig("sample_check.png", dpi=120)
    print("Saved sample_check.png")