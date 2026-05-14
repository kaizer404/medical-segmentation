"""
train.py  (beginner-friendly version)
--------------------------------------
Trains a U-Net on the Brain MRI dataset.
Works on RTX 3050 thanks to mixed precision (AMP).

Run:  python train.py
"""

import time
import json
from pathlib import Path

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import get_dataloaders
from augmentation import get_train_transform, get_val_transform
from model import build_unet,build_attention_unet, CombinedLoss, dice_score, iou_score


# ──────────────────────────────────────────────
#  Settings  (edit these, not the code below)
# ──────────────────────────────────────────────
CFG = {
    "data_root":   "../data/raw/kaggle_3m",
    "img_size":    256,
    "batch_size":  8,           # lower this to 4 if you get an out-of-memory error
    "num_workers": 2,

    "encoder":     "resnet34",
    "pretrained":  "imagenet",

    "epochs":      50,
    "lr":          3e-4,        # same as 0.0003
    "weight_decay": 1e-4,
    "grad_clip":   1.0,         # clips very large gradient updates (prevents instability)

    "save_dir":    "../checkpoints_attention",
    "save_every":  5,           # save a checkpoint every N epochs
}


# ──────────────────────────────────────────────
#  Train for one epoch
# ──────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()   # tells the model "we're training" (enables dropout etc.)
    
    total_loss = total_dice = total_iou = 0.0

    # tqdm wraps the loader to show a progress bar
    for images, masks in tqdm(loader, desc="  Train", leave=False):
        
        # Move data to GPU
        images = images.to(device)
        masks  = masks.to(device)

        # Always clear old gradients before a new step
        optimizer.zero_grad()

        # --- Forward pass (with memory-saving float16) ---
        with autocast():
            predictions = model(images)
            loss = criterion(predictions, masks)

        # --- Backward pass (figure out what to adjust) ---
        scaler.scale(loss).backward()
        
        # Gradient clipping: stops any single update from being too large
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        
        # --- Update the model weights ---
        scaler.step(optimizer)
        scaler.update()

        # Record metrics (no gradient tracking needed here)
        with torch.no_grad():
            dice = dice_score(predictions, masks)
            iou  = iou_score(predictions, masks)

        total_loss += loss.item()
        total_dice += dice.item()
        total_iou  += iou.item()

    # Return averages across all batches
    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


# ──────────────────────────────────────────────
#  Validate for one epoch (no weight updates)
# ──────────────────────────────────────────────
@torch.no_grad()   # decorator that disables gradient tracking for the whole function
def val_epoch(model, loader, criterion, device):
    model.eval()    # tells the model "we're evaluating" (disables dropout etc.)
    
    total_loss = total_dice = total_iou = 0.0

    for images, masks in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device)
        masks  = masks.to(device)

        with autocast():
            predictions = model(images)
            loss = criterion(predictions, masks)

        total_loss += loss.item()
        total_dice += dice_score(predictions, masks).item()
        total_iou  += iou_score(predictions, masks).item()

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────
def main():
    # --- Pick GPU if available, otherwise CPU ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU — training will be very slow!")

    # --- Load the dataset ---
    loaders = get_dataloaders(
        data_root=CFG["data_root"],
        train_transform=get_train_transform(CFG["img_size"]),
        val_transform=get_val_transform(CFG["img_size"]),
        img_size=CFG["img_size"],
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"],
    )
    print(f"\nBatches — train: {len(loaders['train'])}  val: {len(loaders['val'])}\n")

    # --- Build the model ---
    model = build_attention_unet(
        encoder_name=CFG["encoder"],
        encoder_weights=CFG["pretrained"],
    ).to(device)

    # --- Loss function, optimizer, scheduler ---
    criterion = CombinedLoss(ftl_weight=0.7, bce_weight=0.3)
    
    # AdamW is like the standard Adam optimizer but with better weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
    )
    
    # Scheduler: slowly decreases learning rate over time (cosine curve)
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG["epochs"], eta_min=1e-6)
    
    # Scaler: needed for mixed precision (AMP) to work correctly
    scaler = GradScaler()

    # --- Folder to save models ---
    save_dir = Path(CFG["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    best_dice = 0.0
    history   = []      # tracks metrics each epoch so you can plot later

    print(f"Starting training for {CFG['epochs']} epochs...\n")

    # ─── The main training loop ───
    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()

        # Train, then validate
        train_loss, train_dice, train_iou = train_epoch(
            model, loaders["train"], optimizer, criterion, scaler, device
        )
        val_loss, val_dice, val_iou = val_epoch(
            model, loaders["val"], criterion, device
        )
        
        # Step the scheduler (adjusts learning rate)
        scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        # Print a summary line for this epoch
        print(
            f"Epoch [{epoch:02d}/{CFG['epochs']}]  "
            f"Train Loss: {train_loss:.4f}  Dice: {train_dice:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  Dice: {val_dice:.4f}  |  "
            f"LR: {current_lr:.2e}  ({elapsed:.1f}s)"
        )

        # Save metrics to history list
        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 5),
            "train_dice": round(train_dice, 5),
            "val_loss":   round(val_loss,   5),
            "val_dice":   round(val_dice,   5),
        })

        # Save if this is the best model so far
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_dice":    val_dice,
            }, save_dir / "best_model.pth")
            print(f"  ★ Best so far! Val Dice: {best_dice:.4f} → saved best_model.pth")

        # Periodic checkpoint every N epochs
        if epoch % CFG["save_every"] == 0:
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_dice":    val_dice,
            }, save_dir / f"checkpoint_epoch{epoch:02d}.pth")

        # Show how much GPU memory is being used
        if device.type == "cuda":
            used_gb = torch.cuda.memory_reserved(0) / 1e9
            print(f"  VRAM in use: {used_gb:.2f} GB")

        print()  # blank line between epochs

    # Save full history to a JSON file (useful for plotting a loss curve later)
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Done! Best Val Dice: {best_dice:.4f}")
    print(f"Files saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  # needed on Windows
    main()