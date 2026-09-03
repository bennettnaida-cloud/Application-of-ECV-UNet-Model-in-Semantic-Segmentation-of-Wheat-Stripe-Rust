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

## Code overview

- `EI/run-wheat.py` is the single training and evaluation entry point. It builds the model, checks image-mask pairs, creates a reproducible train/validation split, trains with BCE+Dice loss, reports Dice/IoU/accuracy/precision/recall/F1, saves the best checkpoint, and can export validation predictions.
- `EI/Data_Loader.py` pairs files by filename stem, applies the same geometric transforms to each image and its mask, and converts masks to binary tensors.
- `EI/losses.py` contains the BCE+Dice objective and threshold helpers.
- `EI/Metrics.py` contains pixel-level segmentation metrics.
- `EI/Models.py` contains the baseline UNet family; `EI/model/` contains ECV-UNet and other comparison architectures.

## Usage

Install the dependencies and run from the repository root:

```bash
pip install -r EI/requirements.txt
python EI/run-wheat.py
```

The default run uses the paper's ECV-UNet architecture, a deterministic 80/20 split, `96×96` inputs (matching the original loader configuration), 300 epochs, batch size 16, SGD with learning rate `1e-4`, and automatic CPU/GPU selection. To use the paper's stated `256×256` input size:

```bash
python EI/run-wheat.py
```

Each run creates a timestamped directory under `EI/results/<model>/` containing `best_model.pth`, `training.jsonl`, TensorBoard scalars (when `tensorboardX` is installed), and optional predictions.

### Command-line parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `--model` | `ECV-UNet` | Architecture. `ECV-UNet` is the paper model (`CBAM_ECAVUnet`); `U_Net`, `ECAUnet`, `CBAM_ECAUNet`, and `CBAM_ECAVUnet` are also available. |
| `--data-root` | `data` | Directory containing `images/` and `masks/`. |
| `--results-dir` | `EI/results` | Parent directory for checkpoints, logs, and predictions. |
| `--gpu` | `0` | CUDA device index. CPU is selected automatically when CUDA is unavailable or the index is invalid. |
| `--image-size` | `96` | Images and masks are resized to this square size before training. Use `256` for the paper setting. |
| `--epochs` | `300` | Maximum training epochs. |
| `--batch-size` | `16` | Number of samples per optimizer step. |
| `--valid-size` | `0.2` | Fraction reserved for validation/evaluation. |
| `--lr` | `0.0001` | Initial SGD learning rate. |
| `--momentum` | `0.99` | SGD momentum. |
| `--weight-decay` | `0` | L2 regularization coefficient. |
| `--patience` | `300` | Epochs without validation-loss improvement before early stopping. |
| `--threshold` | `0.5` | Sigmoid probability threshold used to form binary masks and metrics. |
| `--workers` | `0` | DataLoader worker processes; `0` is safest on Windows. |
| `--seed` | `42` | Seed for the split and random operations. |
| `--augment` | off | Enables synchronized horizontal flip, vertical flip, and rotation during training. |
| `--save-predictions` | off | Saves binary validation predictions in the run directory. |

For a quick smoke test, use a small epoch count and no worker processes:

```bash
python EI/run-wheat.py --model U_Net --epochs 1 --batch-size 2 --workers 0
```
