"""Train and evaluate the ECV-UNet wheat stripe-rust segmentation model.

Run from the repository root with ``python EI/run-wheat.py``.  The script
keeps the default 96x96 input used by the original loader, while allowing the
paper's 256x256 setting through ``--image-size 256``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

# Some Windows PyTorch/NumPy installations load two OpenMP runtimes. Keep
# the original Windows workaround before importing numeric libraries.
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Subset

from Data_Loader import Images_Dataset_folder
from losses import calc_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a wheat stripe-rust segmentation model."
    )
    parser.add_argument(
        "--model",
        choices=("ECV-UNet", "U_Net", "ECAUnet", "CBAM_ECAUNet", "CBAM_ECAVUnet"),
        default="ECV-UNet",
        help="Model architecture. ECV-UNet maps to CBAM_ECAVUnet (default).",
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "EI" / "results")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index; CPU is used if unavailable.")
    parser.add_argument("--image-size", type=int, default=96, help="Square input size (default: 96).")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--valid-size", type=float, default=0.2, help="Validation fraction in [0, 1).")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial SGD learning rate.")
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=300, help="Early-stopping patience in epochs.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binary masks.")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers; 0 is safest on Windows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(name: str) -> torch.nn.Module:
    if name == "ECV-UNet":
        from model.CBAM_ECAVUnet import CBAM_ECAVUnet

        return CBAM_ECAVUnet(3, 1)
    if name == "U_Net":
        from Models import U_Net

        return U_Net(3, 1)
    if name == "ECAUnet":
        from model.ECAUnet import ECAUnet

        return ECAUnet(3, 1)
    if name == "CBAM_ECAUNet":
        from model.CBAM_ECAUNet import CBAM_ECAUNet

        return CBAM_ECAUNet(3, 1)
    if name == "CBAM_ECAVUnet":
        from model.CBAM_ECAVUnet import CBAM_ECAVUnet

        return CBAM_ECAVUnet(3, 1)
    raise ValueError(f"Unknown model: {name}")


def make_transforms(image_size: int, augment: bool):
    image_ops = [torchvision.transforms.Resize((image_size, image_size))]
    mask_ops = [
        torchvision.transforms.Resize(
            (image_size, image_size), interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )
    ]
    if augment:
        image_ops.extend(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomRotation(15),
            ]
        )
        mask_ops.extend(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomRotation(
                    15, interpolation=torchvision.transforms.InterpolationMode.NEAREST
                ),
            ]
        )
    image_ops.extend(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    mask_ops.extend([torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor()])
    return torchvision.transforms.Compose(image_ops), torchvision.transforms.Compose(mask_ops)


def evaluate(model, loader, device, threshold: float) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            loss = calc_loss(logits, masks)
            loss_sum += loss.item() * images.size(0)
            sample_count += images.size(0)
            predictions = torch.sigmoid(logits) >= threshold
            targets = masks >= 0.5
            tp += torch.logical_and(predictions, targets).sum().item()
            fp += torch.logical_and(predictions, ~targets).sum().item()
            tn += torch.logical_and(~predictions, ~targets).sum().item()
            fn += torch.logical_and(~predictions, targets).sum().item()

    eps = 1e-7
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = 100 * (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {
        "loss": loss_sum / max(sample_count, 1),
        "dice": dice,
        "iou": iou,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_predictions(model, dataset, indices, output_dir: Path, device, threshold: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for index in indices:
            image, _ = dataset[index]
            prediction = torch.sigmoid(model(image.unsqueeze(0).to(device)))[0, 0]
            mask = (prediction >= threshold).cpu().numpy().astype(np.uint8) * 255
            stem = dataset.pairs[index][0].stem
            Image.fromarray(mask).save(output_dir / f"{stem}.png")


def main() -> None:
    args = parse_args()
    if not 0 <= args.valid_size < 1:
        raise ValueError("--valid-size must be in [0, 1)")
    if args.image_size <= 0 or args.batch_size <= 0 or args.epochs <= 0:
        raise ValueError("image size, batch size, and epochs must be positive")
    if args.gpu < 0 or args.workers < 0:
        raise ValueError("--gpu and --workers must be non-negative")
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be in (0, 1)")

    set_seed(args.seed)
    if torch.cuda.is_available() and args.gpu < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    data_root = args.data_root.expanduser().resolve()
    image_dir = data_root / "images"
    mask_dir = data_root / "masks"
    eval_image_transform, eval_mask_transform = make_transforms(args.image_size, augment=False)
    train_image_transform, train_mask_transform = make_transforms(args.image_size, augment=args.augment)
    train_dataset = Images_Dataset_folder(image_dir, mask_dir, train_image_transform, train_mask_transform)
    eval_dataset = Images_Dataset_folder(image_dir, mask_dir, eval_image_transform, eval_mask_transform)
    if len(train_dataset) < 2:
        raise ValueError("At least two paired samples are required")

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(train_dataset))
    valid_count = max(1, int(round(len(indices) * args.valid_size)))
    valid_count = min(valid_count, len(indices) - 1)
    valid_indices = indices[:valid_count].tolist()
    train_indices = indices[valid_count:].tolist()
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        Subset(eval_dataset, valid_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.model).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model: {args.model} ({parameter_count:,} parameters)")
    print(f"Samples: {len(train_indices)} train / {len(valid_indices)} validation")

    run_dir = args.results_dir.expanduser().resolve() / args.model / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best_model.pth"
    log_path = run_dir / "training.jsonl"
    writer = None
    try:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    except ImportError:
        print("tensorboardX is not installed; continuing without TensorBoard logs.")

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    best_loss = float("inf")
    stale_epochs = 0

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss_sum = 0.0
            start = time.time()
            for images, masks in train_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = calc_loss(model(images), masks)
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item() * images.size(0)
            scheduler.step()

            train_loss = train_loss_sum / len(train_indices)
            metrics = evaluate(model, valid_loader, device, args.threshold)
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"valid_{key}": value for key, value in metrics.items()},
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            if writer is not None:
                writer.add_scalar("loss/train", train_loss, epoch)
                for key, value in metrics.items():
                    writer.add_scalar(f"validation/{key}", value, epoch)
            print(
                f"Epoch {epoch:03d}/{args.epochs} | train loss {train_loss:.5f} | "
                f"valid loss {metrics['loss']:.5f} | Dice {metrics['dice']:.4f} | "
                f"IoU {metrics['iou']:.4f} | {time.time() - start:.1f}s"
            )

            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                stale_epochs = 0
                torch.save(model.state_dict(), best_path)
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break
    finally:
        if writer is not None:
            writer.close()

    if not best_path.exists():
        raise RuntimeError("No checkpoint was saved; validation did not complete")
    try:
        checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    except TypeError:  # compatibility with older PyTorch releases
        checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint)
    final_metrics = evaluate(model, valid_loader, device, args.threshold)
    print("Best validation metrics:", json.dumps(final_metrics, ensure_ascii=False))
    if args.save_predictions:
        save_predictions(model, eval_dataset, valid_indices, run_dir / "predictions", device, args.threshold)
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
