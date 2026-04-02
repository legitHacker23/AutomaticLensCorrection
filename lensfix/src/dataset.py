"""Paired training dataset and test dataset with auto-detection of directory layout."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

# ── layout detection ─────────────────────────────────────────────────

LayoutKind = Literal["A", "B", "C"]

ORIGINAL_NAMES = ("original", "distorted", "input")
GENERATED_NAMES = ("generated", "corrected", "gt", "target")


def _find_subdir(root: Path, candidates: Tuple[str, ...]) -> Optional[str]:
    """Return the first candidate that exists as a subdirectory of *root*."""
    for name in candidates:
        if (root / name).is_dir():
            return name
    return None


def detect_layout(train_dir: str | Path) -> Tuple[LayoutKind, str, str]:
    """Auto-detect whether the training data follows layout A, B, or C.

    Layout A (per-ID folders):
        train_dir/<id>/original.jpg  +  train_dir/<id>/generated.jpg
    Layout B (split folders):
        train_dir/original/<id>.jpg  +  train_dir/generated/<id>.jpg
    Layout C (flat paired files):
        train_dir/<uuid>_original.jpg  +  train_dir/<uuid>_generated.jpg

    Returns (layout_kind, original_key, generated_key).
    """
    root = Path(train_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Training directory not found: {root}")

    # Layout B: subdirectories named original/generated
    orig_key = _find_subdir(root, ORIGINAL_NAMES)
    gen_key = _find_subdir(root, GENERATED_NAMES)
    if orig_key is not None and gen_key is not None:
        return "B", orig_key, gen_key

    # Layout A: per-ID subdirectories with original.*/generated.* files
    first_child = next((p for p in sorted(root.iterdir()) if p.is_dir()), None)
    if first_child is not None:
        for oname in ORIGINAL_NAMES:
            matches = list(first_child.glob(f"{oname}.*"))
            if matches:
                gname = None
                for gn in GENERATED_NAMES:
                    if list(first_child.glob(f"{gn}.*")):
                        gname = gn
                        break
                if gname is not None:
                    return "A", oname, gname

    # Layout C: flat files ending with _original.<ext> and _generated.<ext>
    has_orig = any(
        f.is_file() and _is_image(f) and f.stem.endswith("_original")
        for f in root.iterdir()
    )
    has_gen = any(
        f.is_file() and _is_image(f) and f.stem.endswith("_generated")
        for f in root.iterdir()
    )
    if has_orig and has_gen:
        return "C", "_original", "_generated"

    raise RuntimeError(
        f"Cannot detect data layout in {root}. "
        "Expected layout A (<id>/original.* + <id>/generated.*), "
        "layout B (original/<id>.* + generated/<id>.*), or "
        "layout C (<uuid>_original.jpg + <uuid>_generated.jpg)."
    )


# ── pair discovery ───────────────────────────────────────────────────

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def discover_pairs(
    train_dir: str | Path,
) -> Tuple[List[Tuple[Path, Path]], LayoutKind]:
    """Return a sorted list of (original_path, generated_path) pairs."""
    root = Path(train_dir)
    layout, orig_key, gen_key = detect_layout(root)

    pairs: List[Tuple[Path, Path]] = []

    if layout == "C":
        for p in sorted(root.iterdir()):
            if not (p.is_file() and _is_image(p) and p.stem.endswith("_original")):
                continue
            gen_name = p.name.replace("_original", "_generated", 1)
            g = root / gen_name
            if g.exists():
                pairs.append((p, g))
    elif layout == "B":
        orig_dir = root / orig_key
        gen_dir = root / gen_key
        for p in sorted(orig_dir.iterdir()):
            if not _is_image(p):
                continue
            g = gen_dir / p.name
            if not g.exists():
                stem_matches = list(gen_dir.glob(f"{p.stem}.*"))
                g = stem_matches[0] if stem_matches else None
            if g is None or not g.exists():
                raise FileNotFoundError(
                    f"Missing generated pair for {p.name} in {gen_dir}"
                )
            pairs.append((p, g))
    else:
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            orig_files = [f for f in d.iterdir() if f.stem == orig_key and _is_image(f)]
            gen_files = [f for f in d.iterdir() if f.stem == gen_key and _is_image(f)]
            if not orig_files:
                continue
            if not gen_files:
                raise FileNotFoundError(
                    f"Found {orig_key} but no {gen_key} in {d}"
                )
            pairs.append((orig_files[0], gen_files[0]))

    if len(pairs) == 0:
        raise RuntimeError(f"No image pairs found in {root} (layout {layout}).")
    return pairs, layout


# ── photometric augmentation (input only, no geometry change) ────────

class PhotoAugment:
    """Random brightness / contrast jitter applied only to the distorted input."""

    def __init__(self, brightness: float = 0.15, contrast: float = 0.15):
        self.jitter = T.ColorJitter(brightness=brightness, contrast=contrast)

    def __call__(self, img: Image.Image) -> Image.Image:
        return self.jitter(img)


# ── train dataset ────────────────────────────────────────────────────

class LensFixTrainDataset(Dataset):
    """Yields (distorted, corrected) tensor pairs in [0,1] CHW float32.

    Both images are resized identically (deterministic, no random geometry).
    Optional photometric augmentation is applied *only* to the distorted input.
    """

    def __init__(
        self,
        train_dir: str | Path,
        image_size: int = 256,
        augment: bool = True,
        pairs: Optional[List[Tuple[Path, Path]]] = None,
    ):
        if pairs is not None:
            self.pairs = pairs
        else:
            self.pairs, _ = discover_pairs(train_dir)
        print(f"[LensFixTrainDataset] Loaded {len(self.pairs)} pairs from {train_dir}")
        self.image_size = image_size
        self.photo_aug = PhotoAugment() if augment else None

    # ── helpers ───────────────────────────────────────────────────────

    def _load_and_resize(self, path: Path) -> Image.Image:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = self.image_size / min(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
        return TF.center_crop(img, self.image_size)

    # ── interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        orig_path, gen_path = self.pairs[idx]

        x = self._load_and_resize(orig_path)
        y = self._load_and_resize(gen_path)

        if self.photo_aug is not None:
            x = self.photo_aug(x)

        x = TF.to_tensor(x)  # [0,1] CHW float32
        y = TF.to_tensor(y)
        return x, y


# ── test dataset ─────────────────────────────────────────────────────

class LensFixTestDataset(Dataset):
    """Yields (image_tensor, image_id, (H, W)) for each test image.

    image_tensor is [0,1] CHW float32 at *image_size* resolution.
    original_size is the (H, W) tuple of the raw file so inference can
    reconstruct output at the correct resolution.
    """

    def __init__(self, test_dir: str | Path, image_size: int = 256):
        root = Path(test_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Test directory not found: {root}")

        self.paths: List[Path] = sorted(
            p for p in root.iterdir() if _is_image(p)
        )
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {root}.")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, Tuple[int, int]]:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        original_size = (img.height, img.width)

        img_resized = img.resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        tensor = TF.to_tensor(img_resized)
        return tensor, path.name, original_size
