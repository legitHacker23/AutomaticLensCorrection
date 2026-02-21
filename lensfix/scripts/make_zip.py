"""Zip the output directory for Kaggle submission."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def make_zip(input_dir: str, output_path: str) -> Path:
    src = Path(input_dir)
    dst = Path(output_path)
    if not src.is_dir():
        raise FileNotFoundError(f"Input directory not found: {src}")

    images = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise RuntimeError(f"No images found in {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as zf:
        for img in images:
            zf.write(img, img.name)

    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"Created {dst}  ({len(images)} files, {size_mb:.1f} MB)")
    return dst


def main():
    parser = argparse.ArgumentParser(description="Zip output for submission")
    parser.add_argument("--input", default="output",
                        help="Directory containing corrected images")
    parser.add_argument("--output", default="submission.zip",
                        help="Output zip file path")
    args = parser.parse_args()
    make_zip(args.input, args.output)


if __name__ == "__main__":
    main()
