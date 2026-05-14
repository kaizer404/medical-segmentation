"""
app.py
------
Gradio web app for Brain MRI Glioma Segmentation.
Place this file in src/ alongside best_model.pth

Run:
    python app.py
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import segmentation_models_pytorch as smp


# ======================================================================
#  Settings
# ======================================================================

CHECKPOINT = Path(__file__).parent / "best_model.pth"
IMG_SIZE   = 256
N_MC_RUNS  = 20
DROPOUT_P  = 0.1
THRESHOLD  = 0.5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
#  Model helpers
# ======================================================================

def add_dropout(model, p=0.1):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            setattr(model, name, nn.Sequential(module, nn.Dropout2d(p=p)))
        else:
            add_dropout(module, p)
    return model


def enable_dropout(model):
    """Keep dropout ON during inference (MC Dropout trick)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


# ======================================================================
#  Load model once at startup
# ======================================================================

print(f"Loading model on {DEVICE}...")

_model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
)

if CHECKPOINT.exists():
    ckpt = torch.load(str(CHECKPOINT), map_location=DEVICE, weights_only=False)
    _model.load_state_dict(ckpt["model_state"])
    print(f"Checkpoint loaded — Val Dice: {ckpt.get('val_dice', 'N/A')}")
else:
    print(f"WARNING: {CHECKPOINT} not found — using random weights")

_model = add_dropout(_model, p=DROPOUT_P)
_model = _model.to(DEVICE)
_model.eval()

print("Model ready.\n")


# ======================================================================
#  Inference
# ======================================================================

def preprocess(image_np):
    img = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE))
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(DEVICE)


@torch.no_grad()
def mc_inference(tensor):
    enable_dropout(_model)
    preds = []
    for _ in range(N_MC_RUNS):
        logit = _model(tensor)
        prob  = torch.sigmoid(logit).cpu().numpy()[0, 0]
        preds.append(prob)
    preds     = np.stack(preds)
    mean_pred = preds.mean(axis=0)
    variance  = preds.var(axis=0)
    return mean_pred, variance


# ======================================================================
#  Visualisation
# ======================================================================

def make_overlay(img_rgb, pred_binary, variance):
    overlay  = img_rgb.copy()
    var_norm = variance / (variance.max() + 1e-8)
    uncertain = (var_norm > 0.3) & (pred_binary == 1)
    overlay[pred_binary == 1] = (
        overlay[pred_binary == 1] * 0.4 + np.array([80, 200, 80]) * 0.6
    ).astype(np.uint8)
    overlay[uncertain] = (
        overlay[uncertain] * 0.3 + np.array([255, 80, 80]) * 0.7
    ).astype(np.uint8)
    return overlay


def make_confidence_figure(mean_pred, variance):
    var_norm   = variance / (variance.max() + 1e-8)
    confidence = 1 - var_norm

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#111111")
    fig.suptitle("MC Dropout Analysis", color="white", fontsize=13, fontweight="bold")

    axes[0].imshow(mean_pred, cmap="hot", vmin=0, vmax=1)
    axes[0].set_title("Prediction Probability", color="white")
    axes[0].axis("off")

    im = axes[1].imshow(confidence, cmap="RdYlGn", vmin=0, vmax=1)
    axes[1].set_title("Confidence (green=certain, red=uncertain)", color="white")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


# ======================================================================
#  Main predict function
# ======================================================================

def predict(image_np):
    if image_np is None:
        return None, None, "Please upload an MRI scan."

    # Prepare input
    tensor = preprocess(image_np)

    # Run MC inference
    mean_pred, variance = mc_inference(tensor)
    pred_binary = (mean_pred > THRESHOLD).astype(np.uint8)

    # Prepare image for overlay
    img_resized = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE))
    if img_resized.ndim == 2:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    elif img_resized.shape[2] == 4:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGBA2RGB)

    overlay  = make_overlay(img_resized, pred_binary, variance)
    conf_fig = make_confidence_figure(mean_pred, variance)

    # Metrics
    tumour_pct    = pred_binary.sum() / pred_binary.size * 100
    var_norm      = variance / (variance.max() + 1e-8)
    mean_conf     = float(1 - var_norm.mean())
    uncertain_pct = float((var_norm > 0.3).mean() * 100)
    detected      = "✅ Tumour Detected" if pred_binary.sum() > 50 else "✅ No Tumour Detected"

    metrics = f"""
## {detected}

| Metric | Value |
|--------|-------|
| Tumour area | {tumour_pct:.2f}% of scan |
| Mean confidence | {mean_conf:.3f} |
| Uncertain boundary pixels | {uncertain_pct:.1f}% |
| MC Dropout runs | {N_MC_RUNS} |

**Legend:**
- 🟢 Green = confident tumour region
- 🔴 Red = uncertain boundary (radiologist should review)
"""
    return overlay, conf_fig, metrics


# ======================================================================
#  Gradio UI
# ======================================================================

with gr.Blocks(title="Brain MRI Glioma Segmentation") as demo:

    gr.HTML("""
    <div style="text-align:center; padding:20px 0 10px;">
        <h1 style="font-size:2rem; font-weight:700; color:#10b981;">
            🧠 Brain MRI Glioma Segmentation
        </h1>
        <p style="color:#6b7280; font-size:0.95rem;">
            ResNet34 UNet++ &nbsp;·&nbsp; MC Dropout Uncertainty
            &nbsp;·&nbsp; Val Dice 0.8526
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Upload Brain MRI Scan (.tif / .png / .jpg)",
                type="numpy",
            )
            run_btn = gr.Button("🔬 Analyse Scan", variant="primary", size="lg")
            gr.Markdown("""
**How to use:**
1. Upload a brain MRI slice
2. Click Analyse Scan
3. Green overlay = tumour (confident)
4. Red overlay = uncertain boundary
            """)

        with gr.Column(scale=1):
            output_overlay = gr.Image(label="Segmentation Overlay")
            output_metrics = gr.Markdown()

    output_confidence = gr.Plot(label="Confidence Analysis")

    run_btn.click(
        fn=predict,
        inputs=[input_image],
        outputs=[output_overlay, output_confidence, output_metrics],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )