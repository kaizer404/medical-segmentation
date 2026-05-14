# 🧠 Brain MRI Glioma Segmentation

> Deep learning pipeline for automated glioma boundary detection using ResNet34 UNet++ with Monte Carlo Dropout uncertainty estimation.

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-green)](https://huggingface.co/spaces/Kaizer404/brain-mri-segmentation)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org)

---

## 🎯 Overview

This project implements an end-to-end medical image segmentation pipeline for detecting low-grade gliomas (LGG) in brain MRI scans. The model not only segments tumour regions but also provides **uncertainty estimates** — highlighting boundary pixels where the model is less confident, which is critical for clinical decision support.

**Clinical framing:** Glioma boundary detection for radiotherapy treatment planning. Uncertain boundaries (shown in red) indicate regions requiring manual radiologist review.

---

## 🏆 Results

| Model | Val Dice | Test Dice | Test IoU | Epochs |
|-------|----------|-----------|----------|--------|
| ResNet34 U-Net | 0.8043 | 0.8003 | 0.7153 | 50 |
| ResNet34 UNet++ | **0.8526** | 0.7915 | 0.7161 | 50 |
| UNet++ + MC Dropout | — | **0.8886** | — | — |

> UNet++ achieves higher validation Dice due to dense skip connections. MC Dropout averaging over 20 forward passes further improves test Dice by ~0.09.

---

## 🏗️ Architecture

```
Input MRI (256×256×3)
        │
   ┌────▼────┐
   │ ResNet34 │  ← Pretrained ImageNet encoder
   │ Encoder  │     extracts features at 5 scales
   └────┬────┘
        │  skip connections
   ┌────▼────┐
   │  UNet++ │  ← Dense decoder with nested connections
   │ Decoder  │     reconstructs spatial detail
   └────┬────┘
        │
   ┌────▼────┐
   │  Output  │  → Binary mask (256×256×1)
   └─────────┘
```

**Loss Function:** Focal Tversky Loss (α=0.7, β=0.3, γ=0.75) + BCE
- α > β penalises missed tumour pixels more than false alarms
- Focal term focuses training on hard boundary pixels

**Uncertainty:** Monte Carlo Dropout
- Run inference 20× with dropout active
- Mean prediction = final segmentation
- Variance = uncertainty map (high = uncertain boundary)

---

## 📊 Visualisations

### Segmentation Overlay
- 🟢 Green = confident tumour region
- 🔴 Red = uncertain boundary (radiologist should review)

### Confidence Map
- Bright = model is certain
- Dark = model is uncertain

---

## 🗂️ Project Structure

```
medical-segmentation/
├── src/
│   ├── dataset.py          # Data loading + patient-level splits
│   ├── augmentation.py     # Albumentations pipeline
│   ├── model.py            # UNet / UNet++ + losses + metrics
│   ├── train.py            # Training loop (AMP + mixed precision)
│   ├── evaluate.py         # Test set evaluation + plots
│   ├── mc_dropout.py       # MC Dropout uncertainty estimation
│   └── app.py              # Gradio web app
├── data/
│   └── raw/
│       └── kaggle_3m/      # LGG MRI dataset
├── checkpoints/            # U-Net checkpoints
├── checkpoints_attention/  # UNet++ checkpoints
├── results/                # Evaluation plots
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone and install
```bash
git clone https://github.com/YOUR_USERNAME/medical-segmentation
cd medical-segmentation
pip install -r requirements.txt
```

### 2. Download dataset
Download the [LGG MRI Segmentation dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) from Kaggle and place it in `data/raw/kaggle_3m/`

### 3. Train
```bash
cd src
python train.py
```

### 4. Evaluate
```bash
python evaluate.py
```

### 5. Run uncertainty estimation
```bash
python mc_dropout.py
```

### 6. Run the app locally
```bash
python app.py
# Open http://localhost:7860
```

---

## ⚙️ Training Details

| Setting | Value |
|---------|-------|
| GPU | NVIDIA RTX 3050 6GB |
| Batch size | 8 |
| Image size | 256×256 |
| Optimizer | AdamW (lr=3e-4) |
| Scheduler | CosineAnnealingLR |
| Mixed precision | AMP (torch.cuda.amp) |
| Encoder warm-up | Frozen for first 5 epochs |

### Augmentations
- Horizontal + vertical flips
- Affine transforms (rotation ±45°, scale ±15%)
- Elastic deformation
- Grid distortion
- CLAHE contrast enhancement
- Gaussian noise + blur

---

## 📦 Dependencies

```
torch>=2.0.0
segmentation-models-pytorch>=0.3.3
albumentations>=1.3.0
gradio>=4.0.0
opencv-python>=4.7.0
matplotlib>=3.7.0
```

---

## 🌐 Live Demo

Try the live app on Hugging Face Spaces:
**[https://huggingface.co/spaces/Kaizer404/brain-mri-segmentation](https://huggingface.co/spaces/Kaizer404/brain-mri-segmentation)**

Upload any brain MRI slice (.tif / .png / .jpg) to get:
- Tumour segmentation overlay
- MC Dropout confidence map
- Uncertainty metrics

---

## 📚 Dataset

**LGG MRI Segmentation** — Mateusz Buda, Ashirbani Saha, Maciej A. Mazurowski
- 3,929 MRI slices from 110 patients
- 1,373 slices with glioma (35%)
- Source: [Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

---

## 🔬 Clinical Context

Low-grade gliomas are slow-growing brain tumours where precise boundary delineation directly impacts radiotherapy planning. Over- or under-segmentation can lead to either insufficient tumour coverage or unnecessary damage to healthy tissue.

The MC Dropout uncertainty map provides radiologists with a visual indicator of where the model is less confident — typically at tumour edges where tissue contrast is low — enabling targeted manual review rather than full re-segmentation.

---

## 📝 Known Limitations

- Trained only on LGG glioma cases — may not generalise to other tumour types
- 2D slice-by-slice segmentation — 3D context not used
- Performance may degrade on MRI scans from different scanners or protocols

---

*Built with PyTorch · segmentation-models-pytorch · Gradio*
