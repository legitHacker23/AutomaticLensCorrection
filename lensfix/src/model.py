"""HybridWarpNet: coarse radial/tangential warp + residual dense flow + alpha gate."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ── grid helpers ─────────────────────────────────────────────────────

def make_base_grid(B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    """Normalised identity grid in [-1, 1].  Returns (B, H, W, 2)."""
    yy = torch.linspace(-1, 1, H, device=device)
    xx = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
    return grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)


def radial_tangential_grid(
    grid: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
) -> torch.Tensor:
    """Apply Brown–Conrady distortion model to a normalised grid.

    All parameter tensors have shape (B, 1, 1, 1) for broadcasting.
    grid: (B, H, W, 2) in [-1, 1].
    Returns warped grid of same shape.
    """
    x = grid[..., 0:1] - cx  # centre-shifted
    y = grid[..., 1:2] - cy
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2

    x_out = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_out = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    x_out = x_out + cx
    y_out = y_out + cy
    return torch.cat([x_out, y_out], dim=-1)


# ── lightweight encoder ─────────────────────────────────────────────

def _build_encoder(backbone: str = "resnet34") -> tuple[nn.Module, int]:
    """Return (feature_extractor, num_features) from a torchvision backbone."""
    if backbone == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = 512
    elif backbone == "resnet34":
        net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        feat_dim = 512
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    layers = list(net.children())[:-2]  # drop avgpool + fc
    encoder = nn.Sequential(*layers)
    return encoder, feat_dim


# ── flow decoder (lightweight) ───────────────────────────────────────

class FlowDecoder(nn.Module):
    """Predict a low-res 2-channel displacement field from encoder features."""

    def __init__(self, in_channels: int, mid_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 2, 3, padding=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)  # (B, 2, h, w) — low-res


# ── HybridWarpNet ────────────────────────────────────────────────────

class HybridWarpNet(nn.Module):
    """Two-stage warp model: coarse parametric + residual dense flow.

    Forward pass:
        1. Encode input image.
        2. Predict per-image lens params (k1,k2,cx,cy,p1,p2) and alpha.
        3. Build identity grid → apply radial/tangential warp → grid_sample (coarse).
        4. Predict residual flow at low-res, upsample, add to coarse grid → grid_sample (fine).
        5. Alpha-gate scales both coarse params and flow so clean images stay untouched.
    """

    PARAM_COUNT = 6  # k1, k2, cx, cy, p1, p2
    FLOW_SCALE = 0.02  # max displacement magnitude in normalised coords

    def __init__(
        self,
        backbone: str = "resnet34",
        flow_channels: int = 64,
        align_corners: bool = True,
        grid_sample_mode: str = "bicubic",
        grid_sample_padding: str = "border",
    ):
        super().__init__()
        self.align_corners = align_corners
        self.gs_mode = grid_sample_mode
        self.gs_padding = grid_sample_padding

        self.encoder, feat_dim = _build_encoder(backbone)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.param_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.PARAM_COUNT),
        )

        self.alpha_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.flow_decoder = FlowDecoder(feat_dim, flow_channels)

        self._init_heads()

    def _init_heads(self):
        """Start near identity warp: zero params, zero flow, low alpha."""
        for head in (self.param_head, self.alpha_head):
            nn.init.zeros_(head[-1 if isinstance(head[-1], nn.Linear) else -2].weight)
            nn.init.zeros_(head[-1 if isinstance(head[-1], nn.Linear) else -2].bias)
        last_conv = self.flow_decoder.net[-1]
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)

    # ── warp utilities ───────────────────────────────────────────────

    def _sample(self, img: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(
            img,
            grid,
            mode=self.gs_mode,
            padding_mode=self.gs_padding,
            align_corners=self.align_corners,
        )

    # ── forward ──────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x: (B, 3, H, W) distorted image in [0, 1].
        Returns:
            y_hat: (B, 3, H, W) corrected image.
            aux: dict with intermediate tensors for loss / debugging.
        """
        B, _, H, W = x.shape

        feat = self.encoder(x)             # (B, C, h, w)
        pooled = self.gap(feat).flatten(1)  # (B, C)

        # ── alpha gate ───────────────────────────────────────────────
        alpha = self.alpha_head(pooled)                 # (B, 1)
        a = alpha.view(B, 1, 1, 1)                     # broadcastable

        # ── coarse parametric warp ───────────────────────────────────
        params = self.param_head(pooled)                # (B, 6)
        params = torch.tanh(params) * 0.15              # bound to [-0.15, 0.15]
        params = params * alpha                          # alpha-scaled  (B, 6)

        k1, k2, cx, cy, p1, p2 = [p.view(B, 1, 1, 1) for p in params.unbind(1)]

        base_grid = make_base_grid(B, H, W, x.device)
        coarse_grid = radial_tangential_grid(base_grid, k1, k2, cx, cy, p1, p2)
        y_coarse = self._sample(x, coarse_grid)

        # ── residual flow ────────────────────────────────────────────
        flow_lr = self.flow_decoder(feat)               # (B, 2, h, w)
        flow_lr = torch.tanh(flow_lr) * self.FLOW_SCALE
        flow_hr = F.interpolate(
            flow_lr, size=(H, W), mode="bilinear", align_corners=self.align_corners
        )                                                # (B, 2, H, W)
        flow_hr = flow_hr * a                            # alpha-scaled

        # flow_hr is (B,2,H,W) → need (B,H,W,2) for grid addition
        residual = flow_hr.permute(0, 2, 3, 1)
        fine_grid = coarse_grid + residual
        fine_grid = fine_grid.clamp(-1, 1)

        y_hat = self._sample(x, fine_grid)

        aux = {
            "alpha": alpha,          # (B, 1)
            "params": params,        # (B, 6)
            "coarse_grid": coarse_grid,
            "fine_grid": fine_grid,
            "flow_hr": flow_hr,      # (B, 2, H, W)  — needed for TV / bending loss
            "y_coarse": y_coarse,
        }
        return y_hat, aux


# ── quick shape check ────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for res in (128, 256, 384):
        model = HybridWarpNet(backbone="resnet34").to(device)
        x = torch.rand(2, 3, res, res, device=device)
        y_hat, aux = model(x)

        assert y_hat.shape == x.shape, f"output shape mismatch at {res}"
        assert aux["alpha"].shape == (2, 1)
        assert aux["params"].shape == (2, 6)
        assert aux["flow_hr"].shape == (2, 2, res, res)
        assert aux["fine_grid"].shape == (2, res, res, 2)
        print(f"[OK] res={res}  y_hat={tuple(y_hat.shape)}  "
              f"alpha={aux['alpha'].mean().item():.3f}")

    print("All shape checks passed.")
