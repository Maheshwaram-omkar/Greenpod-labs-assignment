"""Dataset discovery, auditing, deterministic splits, and image transforms."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = False
DEFAULT_CLASSES = ("blackspot", "canker", "fresh", "grenning")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def discover_images(root: str | Path, classes: Sequence[str] = DEFAULT_CLASSES):
    """Return sorted (path, integer_label) records; reject missing class folders."""
    root = Path(root)
    if not any((root / name).is_dir() for name in classes) and (root / "train").is_dir():
        root = root / "train"
    missing = [name for name in classes if not (root / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing class folders under {root}: {', '.join(missing)}")
    records = []
    for label, name in enumerate(classes):
        records.extend((p, label) for p in sorted((root / name).rglob("*"))
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not records:
        raise ValueError(f"No supported images found in {root}")
    return records


def audit_dataset(root: str | Path, classes: Sequence[str] = DEFAULT_CLASSES):
    """Count usable images, unreadable files, and basic dimension outliers."""
    records = discover_images(root, classes)
    counts = Counter(classes[label] for _, label in records)
    unreadable, dimensions = [], []
    for path, _ in records:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                dimensions.append((path, image.width, image.height))
        except Exception as exc:
            unreadable.append((str(path), str(exc)))
    areas = np.array([w * h for _, w, h in dimensions], dtype=float)
    if len(areas):
        q1, q3 = np.percentile(areas, [25, 75]); iqr = q3 - q1
        outliers = [str(path) for (path, w, h) in dimensions
                    if w * h < q1 - 1.5 * iqr or w * h > q3 + 1.5 * iqr]
    else:
        outliers = []
    return {"counts": dict(counts), "total": len(records), "unreadable": unreadable,
            "dimension_outliers": outliers, "dimensions": dimensions}


def stratified_split(records, test_size=0.15, val_size=0.15, seed=42):
    """Split record indices into train/validation/test, preserving class proportions."""
    y = np.array([label for _, label in records]); idx = np.arange(len(records))
    train_val, test = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=relative_val,
                                  random_state=seed, stratify=y[train_val])
    return train, val, test


def train_validation_split(records, val_size=0.15, seed=42):
    """Split training records into training and validation indices."""
    labels = np.array([label for _, label in records])
    indices = np.arange(len(records))
    return train_test_split(
        indices,
        test_size=val_size,
        random_state=seed,
        stratify=labels,
    )


def load_rgb(path: str | Path, size=(224, 224)) -> np.ndarray:
    """Load an image as RGB float32 in [0, 1], resized to (width, height)."""
    with Image.open(path) as image:
        image = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def torch_datasets(root, size=224, seed=42, classes=DEFAULT_CLASSES):
    """Build datasets from provided train/test folders and split train for validation."""
    from torch.utils.data import Dataset, Subset
    from torchvision import transforms
    root = Path(root)
    train_records = discover_images(root / "train", classes)
    test_records = discover_images(root / "test", classes)
    train_indices, val_indices = train_validation_split(
        train_records, val_size=0.15, seed=seed
    )

    class Base(Dataset):
        def __init__(self, records, transform):
            self.records = records
            self.transform = transform

        def __len__(self):
            return len(self.records)

        def __getitem__(self, i):
            path, label = self.records[i]
            with Image.open(path) as im:
                im = im.convert("RGB")
            return self.transform(im), label

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    eval_transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
        normalize])
    train_transform = transforms.Compose([transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.02),
        transforms.ToTensor(), normalize])
    train_base = Base(train_records, train_transform)
    val_base = Base(train_records, eval_transform)
    test_base = Base(test_records, eval_transform)
    return (
        Subset(train_base, train_indices.tolist()),
        Subset(val_base, val_indices.tolist()),
        test_base,
    ), list(classes)
