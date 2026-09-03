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

The dataset is intentionally flattened into the two requested directories. The original training set contains 511 samples; the original test set contains 127 images and one misplaced image (`lolr(84).jpg`) that was restored from the dataset root, giving 639 paired samples in total. Augmented images are not included.

Result folders, logs, caches, and model weights/checkpoints from the local `EI` directory were excluded. If a model requires pretrained weights, download or provide them separately at runtime.

## Setup

```bash
pip install -r EI/requirements.txt
```

The scripts resolve paths relative to the repository, so they do not depend on the original `/home/...` machine paths. `run_optimized.py` can be started from the repository root, for example:

```bash
python EI/run_optimized.py --data-root data --results-dir EI/results
```

Because the requested layout has no separate train/test subdirectories, pass explicit split directories if you later create a train/test split.
