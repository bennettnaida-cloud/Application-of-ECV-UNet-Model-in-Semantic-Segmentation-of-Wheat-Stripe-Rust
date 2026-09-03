from __future__ import print_function, division

import random
from pathlib import Path

import torch
import torchvision
from PIL import Image
from skimage import io
from torch.utils.data import Dataset


def _natural_key(path: Path):
    """Sort filenames naturally (e.g. image2 before image10)."""
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


class Images_Dataset(Dataset):
    """Dataset wrapper for already prepared image and label path lists."""

    def __init__(self, images_dir, labels_dir, transformI=None, transformM=None):
        self.images_dir = list(images_dir)
        self.labels_dir = list(labels_dir)
        if len(self.images_dir) != len(self.labels_dir):
            raise ValueError("images_dir and labels_dir must have the same length")
        self.transformI = transformI
        self.transformM = transformM

    def __len__(self):
        return len(self.images_dir)

    def __getitem__(self, idx):
        image = io.imread(self.images_dir[idx])
        label = io.imread(self.labels_dir[idx])
        if self.transformI:
            image = self.transformI(image)
        if self.transformM:
            label = self.transformM(label)
        return {"images": image, "labels": label}


class Images_Dataset_folder(torch.utils.data.Dataset):
    """Load paired images and masks from two directories.

    Pairing is performed by filename stem instead of directory listing order,
    and all path joins use ``pathlib`` so both Windows and POSIX paths work.
    The same random seed is applied to image and mask transforms to keep
    geometric augmentation aligned.
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(self, images_dir, labels_dir, transformI=None, transformM=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.images_dir}")
        if not self.labels_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {self.labels_dir}")

        image_files = sorted(
            [p for p in self.images_dir.iterdir() if p.is_file() and p.suffix.lower() in self.IMAGE_EXTENSIONS],
            key=_natural_key,
        )
        mask_files = {
            p.stem: p
            for p in self.labels_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.MASK_EXTENSIONS
        }
        image_stems = {p.stem for p in image_files}
        missing_masks = [p.name for p in image_files if p.stem not in mask_files]
        extra_masks = [p.name for stem, p in mask_files.items() if stem not in image_stems]
        if missing_masks or extra_masks:
            details = []
            if missing_masks:
                details.append(f"missing masks for {missing_masks[:5]}")
            if extra_masks:
                details.append(f"masks without images {extra_masks[:5]}")
            raise ValueError("Image/mask stems do not match: " + "; ".join(details))

        self.pairs = [(image_path, mask_files[image_path.stem]) for image_path in image_files]
        self.transformI = transformI or torchvision.transforms.Compose(
            [
                torchvision.transforms.CenterCrop(96),
                torchvision.transforms.RandomRotation((-10, 10)),
                torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.transformM = transformM or torchvision.transforms.Compose(
            [
                torchvision.transforms.CenterCrop(96),
                torchvision.transforms.RandomRotation((-10, 10)),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor(),
            ]
        )

    @property
    def images(self):
        return [image for image, _ in self.pairs]

    @property
    def labels(self):
        return [mask for _, mask in self.pairs]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        with Image.open(mask_path) as mask_file:
            mask = mask_file.convert("L")

        seed = random.getrandbits(32)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.transformI(image)

        random.seed(seed)
        torch.manual_seed(seed)
        mask = self.transformM(mask)
        mask = (mask > 0.5).float()
        return image, mask
