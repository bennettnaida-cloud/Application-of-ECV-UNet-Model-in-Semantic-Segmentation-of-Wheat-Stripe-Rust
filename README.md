# Application of ECV-UNet Model in Semantic Segmentation of Wheat Stripe Rust

This repository contains the code and the original wheat stripe-rust segmentation dataset used with the paper:

**Application of ECV-UNet Model in Semantic Segmentation of Wheat Stripe Rust**  
DOI: `10.1109/ICSIP65915.2025.11171544`

## Repository layout

```text
data/
  images/   639 original JPG images
  masks/    639 corresponding PNG masks
EI/         training code, model definitions, and requirements
```

The dataset contains 639 paired samples. The paper uses an 80/20 train-test split; this repository keeps the samples in the two flat directories requested for convenient download and reuse.

## About the paper

Wheat stripe rust causes substantial yield and quality losses, while field images are difficult to segment because of uneven illumination, cluttered backgrounds, and irregular lesion shapes. The paper proposes **ECV-UNet**, an improved UNet designed for accurate, practical semantic segmentation in these conditions.

The model combines three complementary components:

- **C-Max Pooling (CMP):** combines CBAM channel-spatial attention with max pooling to suppress background noise while retaining small-lesion and boundary information.
- **ECA-Skip connections (ECAS):** applies lightweight Efficient Channel Attention before feature fusion so shallow features contribute useful information without introducing as much redundancy.
- **ViT Block:** replaces the fifth encoder convolution block to capture long-range semantic dependencies while retaining convolutional local feature extraction in earlier layers.

Training uses a hybrid Binary Cross-Entropy plus Dice loss. On the field-scale wheat stripe-rust dataset, ECV-UNet reaches a Dice score of **0.7779** and an IoU (mIoU) of **0.6365**, improving on the baseline UNet by **4.77%** and **7.79%**, respectively. The results indicate stronger lesion-region awareness and boundary recognition, supporting automated disease identification and more precise agricultural management.

## Setup

```bash
pip install -r EI/requirements.txt
```

The scripts resolve paths relative to the repository, so they do not depend on the original machine paths. The wheat experiment can be started from the repository root with:

```bash
python EI/run-wheat.py
```

Because the requested layout has no separate train/test subdirectories, pass explicit split directories if you later create a train/test split.
