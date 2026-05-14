"""
mc_dropout.py
-------------
Week 5: MC Dropout uncertainty estimation for Brain MRI segmentation.

What it does:
- Runs the same MRI scan through the model 20 times with dropout ON
- Each run gives a slightly different prediction
- Mean of 20 runs = final prediction (better than single run)
- Variance of 20 runs = confidence map (high variance = uncertain)

Clinical framing:
- High confidence at tumour centre = reliable boundary
- Low confidence at tumour edges = "doctor should look here"

Run:
    python mc_dropout.py
"""

import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from torch.cuda.amp import autocast
from tqdm import tqdm

from dataset import build_dataframe, split_dataframe, BrainMRIDataset
from augmentation import get_val_transform
from model import build_attention_unet


# ======================================================================
#  Config
# ======================================================================

CFG = {
    "data_root":   "../data/raw/kaggle_3m",
    "checkpoint":  "../checkpoints_attention/best_model.pth",
    "img_size":    256,
    "n_runs":      20,        # number of forward passes per image
    "threshold":   0.5,
    "n_samples":   8,         # number of slices to visualise
    "output_dir":  "../results_uncertainty",
    "dropout_p":   0.1,       # dropout probability to inject
}


# ======================================================================
#  Step 1: Add dropout to the model
#  The model was trained without dropout, so we inject it now.
#  This is standard practice for MC Dropout inference.
# ======================================================================

def add_dropout(model: nn.Module, p: float = 0.3) -> nn.Module:
    """
    Recursively adds dropout after every Conv2d layer in the decoder.
    Encoder stays unchanged (we only want uncertainty in the output head).
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            # Wrap conv with a sequential: Conv → Dropout
            setattr(model, name, nn.Sequential(module, nn.Dropout2d(p=p)))
        else:
            add_dropout(module, p)   # recurse into child modules
    return model


def enable_dropout(model: nn.Module):
    """
    Sets model to eval mode (freezes BatchNorm) but keeps Dropout active.
    This is the key trick for MC Dropout — normally eval() disables dropout.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            module.train()   # keep dropout ON during inference


# ======================================================================
#  Step 2: MC Dropout inference
#  Run the same image N times, collect predictions
# ======================================================================

@torch.no_grad()
def mc_predict(
    model:     nn.Module,
    image:     torch.Tensor,   # shape: (1, 3, H, W)
    n_runs:    int,
    device:    torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        mean_pred   : average prediction across N runs  (H, W)  float 0-1
        variance    : variance across N runs            (H, W)  float
        all_preds   : all N raw predictions             (N, H, W)
    """
    image = image.to(device)
    enable_dropout(model)

    preds = []
    for _ in range(n_runs):
        with autocast():
            logit = model(image)
        prob = torch.sigmoid(logit).cpu().numpy()[0, 0]   # (H, W)
        preds.append(prob)

    preds      = np.stack(preds, axis=0)    # (N, H, W)
    mean_pred  = preds.mean(axis=0)         # (H, W)
    variance   = preds.var(axis=0)          # (H, W)

    return mean_pred, variance, preds


# ======================================================================
#  Step 3: Visualisation
# ======================================================================

def plot_uncertainty(samples, out_path: str):
    """
    For each sample plots:
    Col 1: MRI scan
    Col 2: Ground truth mask
    Col 3: Mean prediction (MC Dropout)
    Col 4: Confidence map (1 - variance, normalised)
    Col 5: Uncertainty overlay on MRI
    """
    n     = len(samples)
    ncols = 5
    fig   = plt.figure(figsize=(20, 4 * n))
    fig.suptitle(
        "MC Dropout Uncertainty Estimation\n"
        "Bright = Confident  |  Dark = Uncertain (check these boundaries)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    col_titles = ["MRI Scan", "Ground Truth", "MC Mean Pred",
                  "Confidence Map", "Uncertainty Overlay"]

    for row, (img, mask, mean_pred, variance) in enumerate(samples):
        # Normalise variance to 0-1 for display
        var_norm    = variance / (variance.max() + 1e-8)
        confidence  = 1 - var_norm          # high = certain, low = uncertain
        binary_pred = (mean_pred > CFG["threshold"]).astype(np.uint8)

        # Overlay: green = correct, red = uncertain boundary
        overlay = img.copy()
        uncertain_mask = (var_norm > 0.3) & (binary_pred == 1)
        overlay[binary_pred == 1]   = [100, 200, 100]   # green = prediction
        overlay[uncertain_mask]     = [255, 100, 100]   # red   = uncertain

        cols = [img, mask, mean_pred, confidence, overlay]
        cmaps = [None, "Reds", "Reds", "RdYlGn", None]

        for col in range(ncols):
            ax = fig.add_subplot(n, ncols, row * ncols + col + 1)
            if col in [0, 4]:
                ax.imshow(cols[col])
            else:
                ax.imshow(cols[col], cmap=cmaps[col], vmin=0, vmax=1)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, fontweight="bold")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_uncertainty_stats(all_variances: list, out_path: str):
    """Histogram of uncertainty values across all test slices."""
    flat = np.concatenate([v.flatten() for v in all_variances])

    fig, ax = plt.subplots(figsize=(9, 4))
    flat_clean = flat[np.isfinite(flat)]   # remove inf/nan values
    vmax = float(np.percentile(flat_clean, 99))
    ax.hist(flat_clean, bins=60, color="#E91E63", edgecolor="white", alpha=0.85,
        range=(0, vmax))
    ax.axvline(flat.mean(), color="#333", linewidth=2,
               label=f"Mean uncertainty: {flat.mean():.5f}")
    ax.set_xlabel("Variance (uncertainty)")
    ax.set_ylabel("Pixel count")
    ax.set_title("Distribution of MC Dropout Uncertainty (Test Set)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ======================================================================
#  Step 4: Compare single-run vs MC Dropout Dice
# ======================================================================

def compare_single_vs_mc(model, loader, device, n_runs=20, threshold=0.5):
    """
    Shows that averaging N runs improves Dice over a single run.
    """
    single_dices, mc_dices = [], []

    for images, masks in tqdm(loader, desc="  Comparing single vs MC"):
        for i in range(len(images)):
            img  = images[i:i+1].to(device)
            mask = masks[i, 0].numpy()

            # Single run
            model.eval()
            with torch.no_grad():
                with autocast():
                    logit = model(img)
            single_pred = (torch.sigmoid(logit).cpu().numpy()[0, 0] > threshold).astype(float)

            # MC run
            mean_pred, _, _ = mc_predict(model, img, n_runs=n_runs, device=device)
            mc_pred = (mean_pred > threshold).astype(float)

            # Dice
            def dice(p, t):
                return (2 * (p * t).sum() + 1e-6) / (p.sum() + t.sum() + 1e-6)

            single_dices.append(dice(single_pred, mask))
            mc_dices.append(dice(mc_pred, mask))

        if len(single_dices) >= 100:   # sample 100 slices is enough
            break

    print(f"\n  Single-run Dice : {np.mean(single_dices):.4f}")
    print(f"  MC Dropout Dice : {np.mean(mc_dices):.4f}  ({n_runs} runs)")
    return np.mean(single_dices), np.mean(mc_dices)


# ======================================================================
#  Main
# ======================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Device : {device}")
    print(f"  MC runs per image : {CFG['n_runs']}")
    print(f"{'='*60}\n")

    out_dir = Path(CFG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────
    ckpt  = torch.load(CFG["checkpoint"], map_location=device, weights_only=False)
    model = build_attention_unet().to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded checkpoint — Val Dice: {ckpt['val_dice']:.4f}\n")

    # Inject dropout into decoder
    model = add_dropout(model, p=CFG["dropout_p"])
    print(f"  Dropout (p={CFG['dropout_p']}) injected into decoder\n")

    # ── Test data ─────────────────────────────────────────────────────
    df = build_dataframe(CFG["data_root"])
    _, _, test_df = split_dataframe(df)

    test_ds = BrainMRIDataset(
        test_df,
        transform=get_val_transform(CFG["img_size"]),
        img_size=CFG["img_size"],
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=8, shuffle=False, num_workers=2
    )
    print(f"  Test slices: {len(test_ds)}\n")

    # ── Collect samples for visualisation ─────────────────────────────
    print("  Running MC Dropout inference on sample slices...")
    samples      = []
    all_variances = []

    for images, masks in tqdm(test_loader, desc="  MC inference"):
        for i in range(len(images)):
            img_tensor = images[i:i+1]
            mask_np    = masks[i, 0].numpy()
            img_np     = (images[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            mean_pred, variance, _ = mc_predict(
                model, img_tensor, CFG["n_runs"], device
            )
            all_variances.append(variance)

            # Only collect positive samples (with tumour) for visualisation
            if mask_np.sum() > 0 and len(samples) < CFG["n_samples"]:
                samples.append((img_np, mask_np, mean_pred, variance))

        if len(all_variances) >= 200:
            break

    # ── Compare single vs MC Dice ─────────────────────────────────────
    print("\n  Comparing single-run vs MC Dropout Dice...")
    single_dice, mc_dice = compare_single_vs_mc(
        model, test_loader, device, n_runs=CFG["n_runs"]
    )

    # ── Save results ──────────────────────────────────────────────────
    results = {
        "single_run_dice": round(single_dice, 5),
        "mc_dropout_dice": round(mc_dice, 5),
        "n_runs":          CFG["n_runs"],
        "dropout_p":       CFG["dropout_p"],
    }
    with open(out_dir / "uncertainty_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────
    print("\n  Generating plots...")
    plot_uncertainty(samples, str(out_dir / "uncertainty_maps.png"))
    plot_uncertainty_stats(all_variances, str(out_dir / "uncertainty_histogram.png"))

    print(f"\n{'='*60}")
    print(f"  MC Dropout Results")
    print(f"  Single-run Dice : {single_dice:.4f}")
    print(f"  MC Dropout Dice : {mc_dice:.4f}  ({CFG['n_runs']} runs)")
    print(f"  Improvement     : +{(mc_dice - single_dice):.4f}")
    print(f"  Outputs saved to: {out_dir.resolve()}")
    print(f"{'='*60}\n")
    print("  Clinical interpretation:")
    print("  → High variance pixels = uncertain boundaries")
    print("  → These are the pixels a radiologist should review carefully")
    print("  → Low variance = model is confident = reliable prediction\n")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()