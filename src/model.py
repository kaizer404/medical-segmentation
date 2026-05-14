"""
model.py
---------------
Same as model.py but with all the complexity removed.
Easier to read and understand. Produces identical results.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


# ======================================================================
#  1. THE MODEL
#  ResNet34 shrinks the image to find features (encoder)
#  U-Net expands it back to draw the mask (decoder)
# ======================================================================

def build_unet():
    return smp.Unet(
        encoder_name="resnet34",      # pretrained feature extractor
        encoder_weights="imagenet",   # start from ImageNet weights
        in_channels=3,                # RGB input
        classes=1,                    # output: 1 channel (tumour or not)
        activation=None,              # raw numbers out, sigmoid applied later
    )


def build_attention_unet(
    encoder_name:    str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels:     int = 3,
    num_classes:     int = 1,
) -> nn.Module:
    return smp.UnetPlusPlus(
        encoder_name = encoder_name,
        encoder_weights = encoder_weights,
        in_channels = in_channels,
        classes = num_classes,
        activation = None,
    )
    



# ======================================================================
#  2. LOSS FUNCTION
#  Punishes the model more for missing tumour pixels than for
#  false alarms — important because tumours are tiny (class imbalance)
# ======================================================================

class FocalTverskyLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.alpha  = 0.7    # how much to punish missing tumour (FN)
        self.beta   = 0.3    # how much to punish false alarms (FP)
        self.gamma  = 0.75   # focus on hard examples
        self.smooth = 1e-6   # avoid division by zero

    def forward(self, logits, targets):
        probs   = torch.sigmoid(logits).view(-1)  # convert to 0-1
        targets = targets.view(-1)

        TP = (probs * targets).sum()               # correctly found tumour
        FN = (targets * (1 - probs)).sum()         # missed tumour
        FP = ((1 - targets) * probs).sum()         # false alarm

        tversky = (TP + self.smooth) / (
            TP + self.alpha * FN + self.beta * FP + self.smooth
        )
        return (1 - tversky) ** self.gamma


class CombinedLoss(nn.Module):
    """Mixes Focal Tversky + basic BCE for more stable training."""

    def __init__(self, ftl_weight=0.7, bce_weight=0.3):
        super().__init__()

        self.ftl = FocalTverskyLoss()
        self.bce = nn.BCEWithLogitsLoss()

        self.ftl_w = ftl_weight
        self.bce_w = bce_weight

    def forward(self, logits, targets):

        return (
            self.ftl_w * self.ftl(logits, targets)
            + self.bce_w * self.bce(logits, targets)
        )


# ======================================================================
#  3. METRICS
#  How do we measure if the prediction is good?
#  Both Dice and IoU go from 0 (terrible) to 1 (perfect)
# ======================================================================

def dice_score(logits, targets, threshold=0.5):
    """Dice = 2 * overlap / (pred_size + true_size)"""
    probs   = (torch.sigmoid(logits) > threshold).float().view(-1)
    targets = targets.view(-1)
    overlap = (probs * targets).sum()
    return (2 * overlap + 1e-6) / (probs.sum() + targets.sum() + 1e-6)


def iou_score(logits, targets, threshold=0.5):
    """IoU = overlap / union"""
    probs   = (torch.sigmoid(logits) > threshold).float().view(-1)
    targets = targets.view(-1)
    overlap = (probs * targets).sum()
    union   = probs.sum() + targets.sum() - overlap
    return (overlap + 1e-6) / (union + 1e-6)


# ======================================================================
#  Quick test — run this file directly to verify everything works
# ======================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    model  = build_unet().to(device)
    images = torch.randn(4, 3, 256, 256).to(device)   # fake batch of 4 MRI scans
    masks  = torch.randint(0, 2, (4, 1, 256, 256)).float().to(device)

    with torch.no_grad():
        logits = model(images)

    loss = CombinedLoss()(logits, masks)
    dice = dice_score(logits, masks)
    iou  = iou_score(logits, masks)

    print(f"Output shape : {logits.shape}")
    print(f"Loss  : {loss.item():.4f}")
    print(f"Dice  : {dice.item():.4f}")
    print(f"IoU   : {iou.item():.4f}")
    print("All good ✓")