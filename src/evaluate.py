"""
evaluate.py  (beginner-friendly version)
-----------------------------------------
Loads best_model.pth and tests it on unseen data.
Saves:
  - test_metrics.json  (Dice + IoU numbers)
  - training_curves.png
  - predictions.png
  - dice_histogram.png

Run:  python evaluate.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
from tqdm import tqdm

from dataset import build_dataframe, split_dataframe, BrainMRIDataset
from augmentation import get_val_transform
from model import build_unet, dice_score, iou_score
from model import build_unet, build_attention_unet, dice_score, iou_score


# ──────────────────────────────────────────────
#  Settings
# ──────────────────────────────────────────────
CFG = {
    "data_root":  "../data/raw/kaggle_3m",
    "checkpoint": "../checkpoints_attention/best_model.pth",
    "history":    "../checkpoints_attention/history.json",
    "img_size":   256,
    "batch_size": 16,
    "num_workers": 2,
    "threshold":  0.5,       # pixel probability above this = predicted tumour
    "output_dir": "../checkpoints_attention",
    "n_samples":  12,        # how many slices to show in the prediction grid
}


# ──────────────────────────────────────────────
#  Load the saved model from disk
# ──────────────────────────────────────────────
def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_attention_unet().to(device)   # ← change build_unet to build_attention_unet
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded checkpoint — epoch {ckpt['epoch']}  "
          f"Val Dice: {ckpt['val_dice']:.4f}")
    return model


# ──────────────────────────────────────────────
#  Run the model on the test set, collect results
# ──────────────────────────────────────────────
@torch.no_grad()   # no gradient tracking needed — we're just testing
def evaluate_loader(model, loader, device, threshold=0.5):
    all_dice = []
    all_iou  = []
    samples  = []   # store some (image, true_mask, prediction) for plotting

    for images, masks in tqdm(loader, desc="  Evaluating"):
        images = images.to(device)
        masks  = masks.to(device)

        # Forward pass with memory-saving float16
        with autocast():
            logits = model(images)

        # Record metrics for this batch
        all_dice.append(dice_score(logits, masks).item())
        all_iou.append(iou_score(logits, masks).item())

        # Collect some samples for the prediction plot
        if len(samples) < CFG["n_samples"]:
            # sigmoid converts raw logits → probabilities between 0 and 1
            probs = torch.sigmoid(logits).cpu().numpy()
            imgs  = images.cpu().numpy()
            msks  = masks.cpu().numpy()

            for i in range(len(imgs)):
                if len(samples) >= CFG["n_samples"]:
                    break

                # Convert image from (C, H, W) → (H, W, C) for matplotlib
                img_np  = (imgs[i].transpose(1, 2, 0) * 255).astype(np.uint8)
                msk_np  = msks[i, 0]                            # true mask
                pred_np = (probs[i, 0] > threshold).astype(np.uint8)  # binary prediction
                prob_np = probs[i, 0]                           # raw confidence map

                samples.append((img_np, msk_np, pred_np, prob_np))

    # Return the mean Dice and IoU across all batches
    return np.mean(all_dice), np.mean(all_iou), samples


# ──────────────────────────────────────────────
#  Plot 1: training curves (loss and dice over epochs)
# ──────────────────────────────────────────────
def plot_training_curves(history_path, out_path):
    with open(history_path) as f:
        history = json.load(f)

    # Pull out the numbers from the history list
    epochs     = [h["epoch"]      for h in history]
    train_dice = [h["train_dice"] for h in history]
    val_dice   = [h["val_dice"]   for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")

    # Left plot: Dice score over time
    axes[0].plot(epochs, train_dice, label="Train Dice", color="#2196F3", linewidth=2)
    axes[0].plot(epochs, val_dice,   label="Val Dice",   color="#F44336", linewidth=2)
    axes[0].axhline(max(val_dice), color="#F44336", linestyle="--", alpha=0.4,
                    label=f"Best: {max(val_dice):.4f}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Dice Score")
    axes[0].set_title("Dice Score")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 1)

    # Right plot: Loss over time
    axes[1].plot(epochs, train_loss, label="Train Loss", color="#2196F3", linewidth=2)
    axes[1].plot(epochs, val_loss,   label="Val Loss",   color="#F44336", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────
#  Plot 2: side-by-side prediction grid
#  Shows: MRI scan | true mask | prediction | confidence
# ──────────────────────────────────────────────
def plot_predictions(samples, out_path, test_dice, test_iou):
    n_rows = len(samples)
    n_cols = 4   # 4 columns per sample

    fig = plt.figure(figsize=(16, 4 * n_rows))
    fig.suptitle(
        f"Predictions — Dice: {test_dice:.4f}  IoU: {test_iou:.4f}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    column_titles = ["MRI Scan", "Ground Truth", "Prediction", "Confidence"]

    for row, (img, mask, pred, prob) in enumerate(samples):
        panels = [img, mask, pred, prob]
        cmaps  = [None, "Reds", "Reds", "hot"]

        for col in range(n_cols):
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)

            if col == 0:
                ax.imshow(panels[col])          # show colour image normally
            else:
                ax.imshow(panels[col], cmap=cmaps[col], vmin=0, vmax=1)

            # Column titles on the first row only
            if row == 0:
                ax.set_title(column_titles[col], fontsize=11, fontweight="bold")

            # Show per-sample dice under the prediction column
            if col == 2:
                tp = (pred * mask).sum()
                sample_dice = (2 * tp + 1e-6) / (pred.sum() + mask.sum() + 1e-6)
                ax.set_xlabel(f"Dice: {sample_dice:.3f}", fontsize=9)

            ax.axis("off")   # hide the axis ticks/labels

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────
#  Plot 3: histogram of per-slice dice scores
#  Useful for spotting if the model fails on certain slices
# ──────────────────────────────────────────────
def plot_dice_histogram(model, loader, device, out_path, threshold=0.5):
    per_slice_dice = []

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="  Dice histogram"):
            images = images.to(device)
            masks  = masks.to(device)

            with autocast():
                logits = model(images)

            # Threshold probabilities to get binary predictions
            preds = (torch.sigmoid(logits) > threshold).float()

            # Compute dice individually for each image in the batch
            for i in range(len(images)):
                p  = preds[i].view(-1)    # flatten to 1D
                m  = masks[i].view(-1)
                tp = (p * m).sum()
                dice = (2 * tp + 1e-6) / (p.sum() + m.sum() + 1e-6)
                per_slice_dice.append(dice.item())

    # Draw histogram
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(per_slice_dice, bins=40, color="#2196F3", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(per_slice_dice), color="#F44336", linewidth=2,
               label=f"Mean Dice: {np.mean(per_slice_dice):.4f}")
    ax.set_xlabel("Dice Score per Slice")
    ax.set_ylabel("Number of slices")
    ax.set_title("Per-Slice Dice Distribution (Test Set)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────
#  Main — connects everything in order
# ──────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")

    # Create the output folder if it doesn't exist
    out_dir = Path(CFG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load the trained model
    model = load_model(CFG["checkpoint"], device)

    # Step 2: Build the test dataloader
    # (same split logic as training — gets the same held-out test slice)
    df = build_dataframe(CFG["data_root"])
    _, _, test_df = split_dataframe(df)

    test_dataset = BrainMRIDataset(
        test_df,
        transform=get_val_transform(CFG["img_size"]),
        img_size=CFG["img_size"],
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=CFG["batch_size"],
        shuffle=False,      # keep order consistent for reproducibility
        num_workers=CFG["num_workers"],
        pin_memory=True,    # speeds up CPU→GPU transfer
    )
    print(f"Test slices: {len(test_dataset)}\n")

    # Step 3: Run evaluation
    test_dice, test_iou, samples = evaluate_loader(
        model, test_loader, device, CFG["threshold"]
    )

    print(f"\n{'='*40}")
    print(f"  TEST RESULTS")
    print(f"  Dice : {test_dice:.4f}")
    print(f"  IoU  : {test_iou:.4f}")
    print(f"{'='*40}\n")

    # Save the numbers to a file
    metrics = {"test_dice": round(test_dice, 5), "test_iou": round(test_iou, 5)}
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Step 4: Pick samples that actually have a tumour for better visualisations
    # (many MRI slices are normal — those are boring to look at)
    positive_samples = [s for s in samples if s[1].sum() > 0][:8]
    if len(positive_samples) < 4:
        positive_samples = samples[:8]   # fallback if not enough tumour slices

    # Step 5: Generate and save all three plots
    print("Generating plots...")

    plot_training_curves(
        CFG["history"],
        str(out_dir / "training_curves.png"),
    )
    plot_predictions(
        positive_samples,
        str(out_dir / "predictions.png"),
        test_dice, test_iou,
    )
    plot_dice_histogram(
        model, test_loader, device,
        str(out_dir / "dice_histogram.png"),
    )

    print(f"\nAll results saved to: {out_dir.resolve()}\n")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()   # needed on Windows
    main()