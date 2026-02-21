"""Composite loss: L1 + Sobel gradient + SSIM + flow TV + flow bending energy."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── L1 ───────────────────────────────────────────────────────────────

def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


# ── Sobel gradient loss ──────────────────────────────────────────────

_SOBEL_X = torch.tensor(
    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
).view(1, 1, 3, 3) / 4.0

_SOBEL_Y = torch.tensor(
    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
).view(1, 1, 3, 3) / 4.0


def _sobel_edges(img: torch.Tensor) -> torch.Tensor:
    """Return per-pixel gradient magnitude for a (B,C,H,W) image."""
    B, C, H, W = img.shape
    sx = _SOBEL_X.to(img.device, img.dtype).expand(C, -1, -1, -1)
    sy = _SOBEL_Y.to(img.device, img.dtype).expand(C, -1, -1, -1)
    gx = F.conv2d(img, sx, padding=1, groups=C)
    gy = F.conv2d(img, sy, padding=1, groups=C)
    return (gx * gx + gy * gy).sqrt()


def sobel_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(_sobel_edges(pred), _sobel_edges(target))


# ── SSIM (pure-pytorch, no external deps) ────────────────────────────

def _gaussian_kernel_1d(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = (-0.5 * (coords / sigma) ** 2).exp()
    return g / g.sum()


def _gaussian_kernel_2d(size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k1d = _gaussian_kernel_1d(size, sigma, device, dtype)
    k2d = k1d.unsqueeze(1) @ k1d.unsqueeze(0)  # (size, size)
    return k2d.expand(channels, 1, size, size).contiguous()


def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """1 − SSIM (so lower is better, like the other losses)."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    C = pred.shape[1]
    kernel = _gaussian_kernel_2d(window_size, sigma, C, pred.device, pred.dtype)
    pad = window_size // 2

    mu_p = F.conv2d(pred, kernel, padding=pad, groups=C)
    mu_t = F.conv2d(target, kernel, padding=pad, groups=C)

    mu_pp = mu_p * mu_p
    mu_tt = mu_t * mu_t
    mu_pt = mu_p * mu_t

    sigma_pp = F.conv2d(pred * pred, kernel, padding=pad, groups=C) - mu_pp
    sigma_tt = F.conv2d(target * target, kernel, padding=pad, groups=C) - mu_tt
    sigma_pt = F.conv2d(pred * target, kernel, padding=pad, groups=C) - mu_pt

    numer = (2.0 * mu_pt + C1) * (2.0 * sigma_pt + C2)
    denom = (mu_pp + mu_tt + C1) * (sigma_pp + sigma_tt + C2)
    ssim_map = numer / denom

    return 1.0 - ssim_map.mean()


# ── Flow regularisation ──────────────────────────────────────────────

def flow_tv_loss(flow: torch.Tensor) -> torch.Tensor:
    """Anisotropic total-variation on a (B, 2, H, W) flow field."""
    dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean()
    dy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs().mean()
    return dx + dy


def flow_bending_loss(flow: torch.Tensor) -> torch.Tensor:
    """Second-derivative (bending energy) penalty on a (B, 2, H, W) flow field.

    Penalises d²f/dx², d²f/dy², and d²f/dxdy to encourage smooth warps.
    """
    dxx = flow[:, :, :, 2:] - 2.0 * flow[:, :, :, 1:-1] + flow[:, :, :, :-2]
    dyy = flow[:, :, 2:, :] - 2.0 * flow[:, :, 1:-1, :] + flow[:, :, :-2, :]
    dxy = (
        flow[:, :, 1:, 1:] - flow[:, :, 1:, :-1]
        - flow[:, :, :-1, 1:] + flow[:, :, :-1, :-1]
    )
    return (dxx ** 2).mean() + (dyy ** 2).mean() + (dxy ** 2).mean()


# ── Composite loss ───────────────────────────────────────────────────

class CompositeLoss(nn.Module):
    """Weighted combination of all loss terms.

    Returns a dict with individual components and ``total``.
    """

    def __init__(
        self,
        w_l1: float = 1.0,
        w_grad: float = 1.0,
        w_ssim: float = 0.2,
        w_tv: float = 0.1,
        w_bend: float = 0.05,
    ):
        super().__init__()
        self.w_l1 = w_l1
        self.w_grad = w_grad
        self.w_ssim = w_ssim
        self.w_tv = w_tv
        self.w_bend = w_bend

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        flow: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        L_l1 = l1_loss(pred, target)
        L_grad = sobel_gradient_loss(pred, target)
        L_ssim = ssim_loss(pred, target)
        L_tv = flow_tv_loss(flow)
        L_bend = flow_bending_loss(flow)

        total = (
            self.w_l1 * L_l1
            + self.w_grad * L_grad
            + self.w_ssim * L_ssim
            + self.w_tv * L_tv
            + self.w_bend * L_bend
        )

        return {
            "total": total,
            "l1": L_l1,
            "grad": L_grad,
            "ssim": L_ssim,
            "tv": L_tv,
            "bend": L_bend,
        }


# ── quick sanity check ──────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, C, H, W = 2, 3, 64, 64
    pred = torch.rand(B, C, H, W, device=device)
    target = torch.rand(B, C, H, W, device=device)
    flow = torch.randn(B, 2, H, W, device=device) * 0.05

    criterion = CompositeLoss().to(device)
    out = criterion(pred, target, flow)

    for k, v in out.items():
        assert v.shape == (), f"{k} should be scalar"
        assert torch.isfinite(v), f"{k} is not finite"
        print(f"  {k:>6s} = {v.item():.5f}")

    # identical images → near-zero image losses
    out2 = criterion(target, target, flow)
    assert out2["l1"].item() < 1e-6, "L1 of identical images should be ~0"
    assert out2["grad"].item() < 1e-5, "Grad of identical images should be ~0"
    assert out2["ssim"].item() < 1e-4, "SSIM of identical images should be ~0"
    print("  [OK] identical-image sanity checks passed")

    print("All loss checks passed.")
