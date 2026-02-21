"""Inference: load checkpoint, correct test images at original resolution, save."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm

from src.model import HybridWarpNet
from src.utils import load_config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ── model loading ────────────────────────────────────────────────────

def load_model(cfg: dict, checkpoint: str, device: torch.device) -> HybridWarpNet:
    mcfg = cfg["model"]
    model = HybridWarpNet(
        backbone=mcfg["backbone"],
        flow_channels=mcfg["flow_channels"],
        align_corners=mcfg["align_corners"],
        grid_sample_mode=mcfg["grid_sample_mode"],
        grid_sample_padding=mcfg["grid_sample_padding"],
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: {checkpoint}  (epoch {ckpt.get('epoch', '?')})")
    return model


# ── single-image inference ───────────────────────────────────────────

@torch.no_grad()
def correct_image_fullres(
    model: HybridWarpNet,
    img: Image.Image,
    device: torch.device,
) -> Image.Image:
    """Run model directly at the image's native resolution."""
    tensor = TF.to_tensor(img).unsqueeze(0).to(device)  # (1,3,H,W)
    y_hat, _ = model(tensor)
    return _tensor_to_pil(y_hat)


@torch.no_grad()
def correct_image_fixedres(
    model: HybridWarpNet,
    img: Image.Image,
    device: torch.device,
    infer_size: int = 384,
) -> Image.Image:
    """Predict warp at fixed resolution, upsample grid, apply to full-res image."""
    H, W = img.height, img.width
    small = img.resize((infer_size, infer_size), Image.BILINEAR)
    tensor_small = TF.to_tensor(small).unsqueeze(0).to(device)

    _, aux = model(tensor_small)
    grid = aux["fine_grid"]  # (1, h, w, 2) normalised

    grid_up = grid.permute(0, 3, 1, 2)  # (1,2,h,w)
    grid_up = F.interpolate(grid_up, size=(H, W), mode="bilinear",
                            align_corners=model.align_corners)
    grid_up = grid_up.permute(0, 2, 3, 1)  # (1,H,W,2)

    tensor_full = TF.to_tensor(img).unsqueeze(0).to(device)
    y_hat = F.grid_sample(
        tensor_full, grid_up,
        mode=model.gs_mode,
        padding_mode=model.gs_padding,
        align_corners=model.align_corners,
    )
    return _tensor_to_pil(y_hat)


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """(1,3,H,W) float [0,1] → PIL RGB uint8."""
    arr = tensor.squeeze(0).clamp(0, 1).mul(255).byte().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(arr, "RGB")


# ── batch runner ─────────────────────────────────────────────────────

def run_inference(
    cfg: dict,
    checkpoint: str | None = None,
    output_dir: str | None = None,
    full_res: bool = True,
    infer_size: int = 384,
):
    icfg = cfg["infer"]
    checkpoint = checkpoint or icfg["checkpoint"]
    output_dir = Path(output_dir or icfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, checkpoint, device)

    test_dir = Path(cfg["data"]["test_dir"])
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    paths = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not paths:
        raise RuntimeError(f"No images found in {test_dir}")

    mode = "full-res" if full_res else f"fixed-{infer_size}"
    print(f"Inference mode: {mode}  |  {len(paths)} images  |  output → {output_dir}")

    t0 = time.time()
    saved = []
    for path in tqdm(paths, desc="infer"):
        img = Image.open(path).convert("RGB")

        if full_res:
            corrected = correct_image_fullres(model, img, device)
        else:
            corrected = correct_image_fixedres(model, img, device, infer_size)

        assert corrected.size == img.size, (
            f"Size mismatch for {path.name}: {corrected.size} vs {img.size}"
        )
        out_path = output_dir / path.name
        corrected.save(out_path, "JPEG", quality=95)
        saved.append(path.name)

    elapsed = time.time() - t0
    print(f"Saved {len(saved)} images in {elapsed:.1f}s ({elapsed/len(saved):.2f}s/img)")

    # ── verification ─────────────────────────────────────────────────
    expected = {p.name for p in paths}
    actual = {p.name for p in output_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    missing = expected - actual
    extra = actual - expected
    if missing:
        print(f"WARNING: {len(missing)} missing files: {sorted(missing)[:5]}...")
    if extra:
        print(f"WARNING: {len(extra)} extra files: {sorted(extra)[:5]}...")
    if not missing and not extra:
        print(f"Verification OK: {len(actual)} files match test set exactly.")

    return output_dir


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LensFix inference")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-full-res", action="store_true",
                        help="Use fixed-res warp upsampling instead of full-res")
    parser.add_argument("--infer-size", type=int, default=384,
                        help="Resolution for fixed-res mode")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_inference(
        cfg,
        checkpoint=args.checkpoint,
        output_dir=args.output,
        full_res=not args.no_full_res,
        infer_size=args.infer_size,
    )


if __name__ == "__main__":
    main()
