from __future__ import annotations

import argparse
import csv
import glob
import importlib
import json
import os
import random
import re
import sys
import time
import types
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

# Some Windows research environments load multiple OpenMP runtimes through
# NumPy/PyTorch stacks. Set this before importing numeric libraries.
if os.name == "nt" and os.environ.get("EI_DISABLE_OPENMP_WORKAROUND") != "1":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import torchvision
from PIL import Image
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import InterpolationMode

from Data_Loader import Images_Dataset_folder
from losses import calc_loss

try:
    import natsort
except ModuleNotFoundError:
    natsort = None


TIMM_AVAILABLE = importlib.util.find_spec("timm") is not None
if not TIMM_AVAILABLE:
    timm_module = types.ModuleType("timm")
    timm_models_module = types.ModuleType("timm.models")
    timm_vit_module = types.ModuleType("timm.models.vision_transformer")

    class MissingTimmBlock:
        def __init__(self, *args, **kwargs):
            raise ImportError("This model requires the optional dependency 'timm'.")

    timm_vit_module.Block = MissingTimmBlock
    sys.modules.setdefault("timm", timm_module)
    sys.modules.setdefault("timm.models", timm_models_module)
    sys.modules.setdefault("timm.models.vision_transformer", timm_vit_module)

from Models import AttU_Net, NestedUNet, R2AttU_Net, R2U_Net, U_Net


SCRIPT_DIR = Path(__file__).resolve().parent
ORIGINAL_DATA_ROOT = SCRIPT_DIR.parent / "data"
ORIGINAL_RESULTS_DIR = SCRIPT_DIR / "results"


def optional_model(module_name: str, class_name: str):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None
    return getattr(module, class_name, None)


def build_model_registry() -> dict[str, Callable[..., nn.Module]]:
    registry: dict[str, Callable[..., nn.Module]] = {
        "U_Net": U_Net,
        "R2U_Net": R2U_Net,
        "AttU_Net": AttU_Net,
        "R2AttU_Net": R2AttU_Net,
        "NestedUNet": NestedUNet,
    }
    for module_name, class_name, requires_timm in [
        ("model.VIT", "VIT", True),
        ("model.MVIT", "MVIT", True),
        ("model.MVNUnet", "MVNUnet", False),
        ("model.ECAMVNUnet", "ECAMVNUnet", False),
    ]:
        if requires_timm and not TIMM_AVAILABLE:
            continue
        model_class = optional_model(module_name, class_name)
        if model_class is not None:
            registry[class_name] = model_class
    return registry


MODEL_REGISTRY = build_model_registry()


def natural_sorted(values: list[str]) -> list[str]:
    if natsort is not None:
        return natsort.natsorted(values)

    def key(value: str):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", value)]

    return sorted(values, key=key)


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    valid_loss: float
    dice: float
    iou: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    lr: float
    seconds: float


def parse_args() -> argparse.Namespace:
    default_data_root = ORIGINAL_DATA_ROOT
    default_results_dir = ORIGINAL_RESULTS_DIR

    parser = argparse.ArgumentParser(
        description="Optimized training entrypoint for EI segmentation models."
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use.")
    parser.add_argument("--model", default="U_Net", choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--results-dir", type=Path, default=default_results_dir)
    parser.add_argument("--train-images", type=Path, default=None)
    parser.add_argument("--train-masks", type=Path, default=None)
    parser.add_argument("--test-images-glob", type=str, default=None)
    parser.add_argument("--test-masks-glob", type=str, default=None)
    parser.add_argument("--sample-image", type=Path, default=None)
    parser.add_argument("--sample-mask", type=Path, default=None)
    parser.add_argument("--image-size", type=int, nargs=2, default=(256, 256), metavar=("H", "W"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--optimizer", choices=("adamw", "adam", "sgd"), default="adamw")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", choices=("cosine", "none"), default="cosine")
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preview-every", type=int, default=10)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save periodic last_model checkpoints every N epochs; 0 saves only at the end.",
    )
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--save-test-preds", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_paths(args: argparse.Namespace) -> dict[str, Path | str]:
    data_root = args.data_root
    flat_images = data_root / "images"
    flat_masks = data_root / "masks"
    structured = (data_root / "train" / "images").exists()
    train_images = args.train_images or (data_root / "train" / "images" if structured else flat_images)
    train_masks = args.train_masks or (data_root / "train" / "masks" if structured else flat_masks)
    test_images_glob = args.test_images_glob or str(
        data_root / "test" / "images" / "*.jpg" if structured else flat_images / "*.jpg"
    )
    test_masks_glob = args.test_masks_glob or str(
        data_root / "test" / "masks" / "*.png" if structured else flat_masks / "*.png"
    )
    sample_candidates = sorted(Path(train_images).glob("*.jpg"))
    sample_image = args.sample_image or (sample_candidates[0] if sample_candidates else Path(train_images) / "sample.jpg")
    sample_mask = args.sample_mask or Path(train_masks) / (sample_image.stem + ".png")
    return {
        "train_images": train_images,
        "train_masks": train_masks,
        "test_images_glob": test_images_glob,
        "test_masks_glob": test_masks_glob,
        "sample_image": sample_image,
        "sample_mask": sample_mask,
    }


def require_training_paths(paths: dict[str, Path | str]) -> None:
    missing = [
        str(paths[key])
        for key in ("train_images", "train_masks")
        if not Path(paths[key]).exists()
    ]
    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(
            "Training data was not found. Pass --data-root or explicit paths.\n"
            f"Missing:\n  {joined}"
        )


def make_transforms(image_size: tuple[int, int], augment: bool):
    image_ops = [torchvision.transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR)]
    mask_ops = [torchvision.transforms.Resize(image_size, interpolation=InterpolationMode.NEAREST)]

    if augment:
        image_ops.extend(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomRotation(
                    10, interpolation=InterpolationMode.BILINEAR
                ),
            ]
        )
        mask_ops.extend(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomRotation(
                    10, interpolation=InterpolationMode.NEAREST
                ),
            ]
        )

    image_ops.extend(
        [
            torchvision.transforms.ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.15
            )
            if augment
            else torchvision.transforms.Lambda(lambda image: image),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    mask_ops.extend([torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor()])

    valid_image_ops = [
        torchvision.transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    valid_mask_ops = [
        torchvision.transforms.Resize(image_size, interpolation=InterpolationMode.NEAREST),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor(),
    ]
    return (
        torchvision.transforms.Compose(image_ops),
        torchvision.transforms.Compose(mask_ops),
        torchvision.transforms.Compose(valid_image_ops),
        torchvision.transforms.Compose(valid_mask_ops),
    )


def split_indices(length: int, valid_size: float, seed: int) -> tuple[list[int], list[int]]:
    if length < 2:
        raise ValueError("At least two samples are required for train/validation split.")
    if not 0 < valid_size < 1:
        raise ValueError("--valid-size must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    indices = np.arange(length)
    rng.shuffle(indices)
    split = max(1, int(np.floor(valid_size * length)))
    split = min(split, length - 1)
    return indices[split:].tolist(), indices[:split].tolist()


def make_loaders(args: argparse.Namespace, paths: dict[str, Path | str]):
    train_tf, mask_tf, valid_tf, valid_mask_tf = make_transforms(tuple(args.image_size), args.augment)
    train_dataset = Images_Dataset_folder(
        str(paths["train_images"]) + os.sep,
        str(paths["train_masks"]) + os.sep,
        transformI=train_tf,
        transformM=mask_tf,
    )
    valid_dataset = Images_Dataset_folder(
        str(paths["train_images"]) + os.sep,
        str(paths["train_masks"]) + os.sep,
        transformI=valid_tf,
        transformM=valid_mask_tf,
    )
    train_idx, valid_idx = split_indices(len(train_dataset), args.valid_size, args.seed)

    common_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.workers > 0:
        common_kwargs["persistent_workers"] = True
        common_kwargs["prefetch_factor"] = 2

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        shuffle=True,
        generator=generator,
        drop_last=False,
        **common_kwargs,
    )
    valid_loader = DataLoader(
        Subset(valid_dataset, valid_idx),
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )
    return train_loader, valid_loader, train_idx, valid_idx


def select_device(gpu_index: int) -> torch.device:
    if torch.cuda.is_available():
        if 0 <= gpu_index < torch.cuda.device_count():
            torch.cuda.set_device(gpu_index)
            return torch.device(f"cuda:{gpu_index}")
        print(f"Warning: cuda:{gpu_index} is unavailable, falling back to CPU.")
    return torch.device("cpu")


def build_model(name: str, device: torch.device) -> nn.Module:
    model = MODEL_REGISTRY[name](3, 1)
    return model.to(device)


def build_optimizer(args: argparse.Namespace, model: nn.Module):
    if args.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )


def make_run_dir(args: argparse.Namespace, model_name: str) -> dict[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.results_dir / model_name / timestamp
    dirs = {
        "run": run_dir,
        "pred": run_dir / "pred",
        "models": run_dir / "saved_models",
        "plots": run_dir / "result_plot",
        "tensorboard": run_dir / "runs",
        "test_preds": run_dir / "gen_images",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def confusion_from_logits(
    logits: torch.Tensor, target: torch.Tensor, threshold: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = torch.sigmoid(logits) > threshold
    truth = target > 0.5
    tp = (pred & truth).sum(dtype=torch.float64)
    fp = (pred & ~truth).sum(dtype=torch.float64)
    tn = (~pred & ~truth).sum(dtype=torch.float64)
    fn = (~pred & truth).sum(dtype=torch.float64)
    return tp, fp, tn, fn


def metrics_from_confusion(tp: float, fp: float, tn: float, fn: float) -> dict[str, float]:
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "dice": (2 * tp) / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps) * 100,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + eps),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_seen = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp_enabled):
            logits = model(images)
            loss = calc_loss(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total_loss += loss.detach().item() * batch_size
        total_seen += batch_size

    return total_loss / max(total_seen, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    amp_enabled: bool,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_seen = 0
    total_tp = total_fp = total_tn = total_fn = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with autocast(enabled=amp_enabled):
            logits = model(images)
            loss = calc_loss(logits, masks)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_seen += batch_size

        tp, fp, tn, fn = confusion_from_logits(logits, masks, threshold)
        total_tp += tp.item()
        total_fp += fp.item()
        total_tn += tn.item()
        total_fn += fn.item()

    return total_loss / max(total_seen, 1), metrics_from_confusion(
        total_tp, total_fp, total_tn, total_fn
    )


def save_preview(
    model: nn.Module,
    sample_image: Path,
    output_dir: Path,
    image_size: tuple[int, int],
    epoch: int,
    device: torch.device,
) -> None:
    if not sample_image.exists():
        return

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    model.eval()
    with torch.no_grad():
        image = Image.open(sample_image).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        pred = torch.sigmoid(model(tensor)).squeeze().detach().cpu().numpy()
    plt.imsave(output_dir / f"preview_epoch_{epoch:04d}.png", pred, cmap="gray")


def append_csv(path: Path, stats: EpochStats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(stats)))
        if not exists:
            writer.writeheader()
        writer.writerow(asdict(stats))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def make_summary_writer(log_dir: Path):
    try:
        from tensorboardX import SummaryWriter
    except ModuleNotFoundError:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ModuleNotFoundError:

            class SummaryWriter:  # type: ignore[no-redef]
                def __init__(self, *args, **kwargs):
                    print("TensorBoard is unavailable; scalar logging is disabled.")

                def add_scalar(self, *args, **kwargs):
                    return None

                def close(self):
                    return None

    return SummaryWriter(log_dir=str(log_dir))


def to_jsonable(payload: dict) -> dict:
    converted = {}
    for key, value in payload.items():
        if isinstance(value, Path):
            converted[key] = str(value)
        elif isinstance(value, tuple):
            converted[key] = list(value)
        else:
            converted[key] = value
    return converted


def log_to_tensorboard(writer, stats: EpochStats) -> None:
    step = stats.epoch
    writer.add_scalar("Loss/train", stats.train_loss, step)
    writer.add_scalar("Loss/valid", stats.valid_loss, step)
    writer.add_scalar("Metrics/Dice", stats.dice * 100, step)
    writer.add_scalar("Metrics/IoU", stats.iou * 100, step)
    writer.add_scalar("Metrics/Accuracy", stats.accuracy, step)
    writer.add_scalar("Metrics/Precision", stats.precision * 100, step)
    writer.add_scalar("Metrics/Recall", stats.recall * 100, step)
    writer.add_scalar("Metrics/F1", stats.f1 * 100, step)
    writer.add_scalar("LearningRate", stats.lr, step)


def save_plots(plot_dir: Path, history: list[EpochStats]) -> None:
    if not history:
        return

    plot_dir.mkdir(parents=True, exist_ok=True)
    epochs = [item.epoch for item in history]
    series = {
        "loss_plot.png": {
            "title": "Training and Validation Loss",
            "values": [("Training Loss", [item.train_loss for item in history]), ("Validation Loss", [item.valid_loss for item in history])],
            "ylabel": "Loss",
        },
        "dice_plot.png": {"title": "Dice Coefficient", "values": [("Dice", [item.dice for item in history])], "ylabel": "Dice"},
        "iou_plot.png": {"title": "IoU Score", "values": [("IoU", [item.iou for item in history])], "ylabel": "IoU"},
        "accuracy_plot.png": {"title": "Accuracy", "values": [("Accuracy", [item.accuracy for item in history])], "ylabel": "Accuracy (%)"},
        "precision_plot.png": {"title": "Precision", "values": [("Precision", [item.precision for item in history])], "ylabel": "Precision"},
        "recall_plot.png": {"title": "Recall", "values": [("Recall", [item.recall for item in history])], "ylabel": "Recall"},
        "f1_plot.png": {"title": "F1 Score", "values": [("F1", [item.f1 for item in history])], "ylabel": "F1"},
    }

    for filename, config in series.items():
        plt.figure(figsize=(10, 6))
        for label, values in config["values"]:
            plt.plot(epochs, values, label=label)
        plt.title(config["title"])
        plt.xlabel("Epoch")
        plt.ylabel(config["ylabel"])
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / filename)
        plt.close()


def print_stats(stats: EpochStats) -> None:
    print("-" * 88)
    print(
        f"Epoch {stats.epoch:03d} | "
        f"train {stats.train_loss:.6f} | valid {stats.valid_loss:.6f} | "
        f"dice {stats.dice:.4f} | iou {stats.iou:.4f} | "
        f"acc {stats.accuracy:.2f}% | precision {stats.precision:.4f} | "
        f"recall {stats.recall:.4f} | f1 {stats.f1:.4f} | "
        f"lr {stats.lr:.2e} | {stats.seconds:.1f}s"
    )


def load_best_model(model: nn.Module, path: Path, device: torch.device) -> None:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)


@torch.no_grad()
def evaluate_test_set(
    model: nn.Module,
    image_glob: str,
    mask_glob: str,
    output_dir: Path,
    image_size: tuple[int, int],
    device: torch.device,
    threshold: float,
    save_predictions: bool,
) -> dict[str, float] | None:
    image_paths = natural_sorted(glob.glob(image_glob))
    mask_paths = natural_sorted(glob.glob(mask_glob))
    if not image_paths or not mask_paths:
        print("Test images or masks were not found; skipping final test evaluation.")
        return None
    if len(image_paths) != len(mask_paths):
        print(
            f"Warning: test image/mask count mismatch ({len(image_paths)} vs {len(mask_paths)}). "
            "Using matched prefix."
        )

    image_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    mask_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(image_size, interpolation=InterpolationMode.NEAREST),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor(),
        ]
    )

    model.eval()
    total_tp = total_fp = total_tn = total_fn = 0.0
    limit = min(len(image_paths), len(mask_paths))
    if save_predictions:
        output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(limit):
        image = Image.open(image_paths[index]).convert("RGB")
        mask = Image.open(mask_paths[index])
        image_tensor = image_transform(image).unsqueeze(0).to(device)
        mask_tensor = mask_transform(mask).unsqueeze(0).to(device)
        logits = model(image_tensor)
        tp, fp, tn, fn = confusion_from_logits(logits, mask_tensor, threshold)
        total_tp += tp.item()
        total_fp += fp.item()
        total_tn += tn.item()
        total_fn += fn.item()

        if save_predictions:
            pred = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
            plt.imsave(output_dir / f"test_pred_{index:04d}.png", pred, cmap="gray")

    return metrics_from_confusion(total_tp, total_fp, total_tn, total_fn)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    paths = resolve_paths(args)
    require_training_paths(paths)

    device = select_device(args.gpu)
    model = build_model(args.model, device)
    run_dirs = make_run_dir(args, args.model)
    write_json(
        run_dirs["run"] / "config.json",
        to_jsonable(vars(args)) | {k: str(v) for k, v in paths.items()},
    )

    if not args.no_summary:
        try:
            from torchsummary import summary

            summary(model, input_size=(3, args.image_size[0], args.image_size[1]))
        except Exception as exc:
            print(f"Model summary skipped: {exc}")

    train_loader, valid_loader, train_idx, valid_idx = make_loaders(args, paths)
    optimizer = build_optimizer(args, model)
    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
        if args.scheduler == "cosine"
        else None
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)
    writer = make_summary_writer(run_dirs["tensorboard"])

    print(f"Device: {device}")
    print(f"Run directory: {run_dirs['run']}")
    print(f"Train/valid samples: {len(train_idx)}/{len(valid_idx)}")

    best_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history: list[EpochStats] = []
    csv_path = run_dirs["run"] / "training_metrics.csv"
    best_model_path = run_dirs["models"] / "best_model.pth"
    last_model_path = run_dirs["models"] / "last_model.pth"

    try:
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scaler, device, amp_enabled
            )
            valid_loss, metrics = validate(
                model, valid_loader, device, args.threshold, amp_enabled
            )
            if scheduler is not None:
                scheduler.step()

            stats = EpochStats(
                epoch=epoch,
                train_loss=train_loss,
                valid_loss=valid_loss,
                dice=metrics["dice"],
                iou=metrics["iou"],
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
                lr=optimizer.param_groups[0]["lr"],
                seconds=time.time() - start,
            )
            history.append(stats)
            append_csv(csv_path, stats)
            log_to_tensorboard(writer, stats)
            print_stats(stats)

            if args.preview_every > 0 and epoch % args.preview_every == 0:
                save_preview(
                    model,
                    Path(paths["sample_image"]),
                    run_dirs["pred"],
                    tuple(args.image_size),
                    epoch,
                    device,
                )

            if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
                torch.save(model.state_dict(), last_model_path)
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                bad_epochs = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"Validation loss improved; saved {best_model_path}")
            else:
                bad_epochs += 1

            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break
    finally:
        writer.close()
        save_plots(run_dirs["plots"], history)

    if not best_model_path.exists():
        torch.save(model.state_dict(), best_model_path)
    torch.save(model.state_dict(), last_model_path)
    load_best_model(model, best_model_path, device)

    final_payload = {
        "best_epoch": best_epoch,
        "best_valid_loss": best_loss,
        "best_model": str(best_model_path),
        "last_model": str(last_model_path),
    }
    if not args.skip_test:
        test_metrics = evaluate_test_set(
            model,
            str(paths["test_images_glob"]),
            str(paths["test_masks_glob"]),
            run_dirs["test_preds"],
            tuple(args.image_size),
            device,
            args.threshold,
            args.save_test_preds,
        )
        if test_metrics is not None:
            final_payload["test_metrics"] = test_metrics
            print(
                "Test | "
                f"dice {test_metrics['dice']:.4f} | iou {test_metrics['iou']:.4f} | "
                f"acc {test_metrics['accuracy']:.2f}% | "
                f"precision {test_metrics['precision']:.4f} | "
                f"recall {test_metrics['recall']:.4f} | f1 {test_metrics['f1']:.4f}"
            )

    write_json(run_dirs["run"] / "final_results.json", final_payload)
    print(f"Finished. Results saved to: {run_dirs['run']}")


if __name__ == "__main__":
    main()
