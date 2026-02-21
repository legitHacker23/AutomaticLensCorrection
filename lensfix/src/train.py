"""Training loop with AMP, checkpointing, validation, and logging."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.dataset import LensFixTrainDataset, discover_pairs
from src.losses import CompositeLoss
from src.model import HybridWarpNet
from src.utils import count_parameters, load_config, seed_everything

LOG_EVERY = 20  # steps between console log lines


# ── helpers ──────────────────────────────────────────────────────────

def _build_loaders(
    pairs, image_size: int, batch_size: int, val_ratio: float, num_workers: int,
):
    n_val = max(1, int(len(pairs) * val_ratio))
    n_train = len(pairs) - n_val
    train_pairs, val_pairs = random_split(
        pairs, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )

    train_ds = LensFixTrainDataset(
        train_dir="", image_size=image_size, augment=True,
        pairs=[pairs[i] for i in train_pairs.indices],
    )
    val_ds = LensFixTrainDataset(
        train_dir="", image_size=image_size, augment=False,
        pairs=[pairs[i] for i in val_pairs.indices],
    )

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_dl, val_dl


def _save_checkpoint(
    path: Path, model, optimizer, scaler, epoch: int, best_val: float, cfg: dict,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "cfg": cfg,
    }, path)


def _load_checkpoint(path: Path, model, optimizer=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("epoch", 0), ckpt.get("best_val", float("inf"))


def _load_weights_only(path: Path, model):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    return ckpt


# ── single epoch ─────────────────────────────────────────────────────

def _run_epoch(
    model, loader, criterion, optimizer, scaler, device,
    *, use_amp: bool, training: bool,
):
    model.train(training)
    totals = {}
    count = 0
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    pbar = tqdm(loader, desc="train" if training else "val  ", leave=False)
    for step, (x, y) in enumerate(pbar):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with autocast(amp_device, enabled=use_amp):
            y_hat, aux = model(x)
            losses = criterion(y_hat, y, aux["flow_hr"])

        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()

        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        count += 1

        if training and (step + 1) % LOG_EVERY == 0:
            avg = {k: v / count for k, v in totals.items()}
            parts = "  ".join(f"{k}={avg[k]:.4f}" for k in sorted(avg))
            pbar.set_postfix_str(parts)

    return {k: v / max(count, 1) for k, v in totals.items()}


# ── train one stage ──────────────────────────────────────────────────

def train_stage(
    cfg: dict,
    stage_key: str,
    model: HybridWarpNet,
    pairs: list,
    device: torch.device,
    resume_path: str | None = None,
    weights_only: bool = False,
):
    scfg = cfg[stage_key]
    image_size = scfg["image_size"]
    batch_size = scfg["batch_size"]
    epochs = scfg["epochs"]
    lr = scfg["lr"]
    use_amp = scfg.get("amp", True)
    val_ratio = cfg["data"].get("val_ratio", 0.05)
    num_workers = cfg["data"].get("num_workers", 4)
    ckpt_dir = Path(cfg["checkpoint"]["dir"])
    save_every = cfg["checkpoint"].get("save_every", 1)

    print(f"\n{'='*60}")
    print(f"Stage: {stage_key}  |  {image_size}px  |  bs={batch_size}  |  "
          f"lr={lr}  |  epochs={epochs}  |  amp={use_amp}")
    print(f"{'='*60}")

    train_dl, val_dl = _build_loaders(
        pairs, image_size, batch_size, val_ratio, num_workers,
    )
    print(f"Train: {len(train_dl.dataset)} pairs  |  Val: {len(val_dl.dataset)} pairs")

    lcfg = cfg["loss"]
    criterion = CompositeLoss(
        w_l1=lcfg["w_l1"], w_grad=lcfg["w_grad"], w_ssim=lcfg["w_ssim"],
        w_tv=lcfg["w_tv"], w_bend=lcfg["w_bend"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = GradScaler(amp_device, enabled=(use_amp and device.type == "cuda"))

    start_epoch = 0
    best_val = float("inf")

    if resume_path and Path(resume_path).is_file():
        if weights_only:
            print(f"Loading weights (not optimizer/epoch) from {resume_path}")
            _load_weights_only(Path(resume_path), model)
            start_epoch = 0
            best_val = float("inf")
        else:
            print(f"Resuming from {resume_path}")
            start_epoch, best_val = _load_checkpoint(
                Path(resume_path), model, optimizer, scaler
            )
            start_epoch += 1
            print(f"  → continuing at epoch {start_epoch}, best_val={best_val:.5f}")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_metrics = _run_epoch(
            model, train_dl, criterion, optimizer, scaler, device,
            use_amp=use_amp, training=True,
        )

        with torch.no_grad():
            val_metrics = _run_epoch(
                model, val_dl, criterion, optimizer, scaler, device,
                use_amp=use_amp, training=False,
            )

        elapsed = time.time() - t0
        t_parts = "  ".join(f"{k}={v:.4f}" for k, v in sorted(train_metrics.items()))
        v_parts = "  ".join(f"{k}={v:.4f}" for k, v in sorted(val_metrics.items()))
        print(f"[{stage_key} ep {epoch+1}/{epochs}  {elapsed:.0f}s]  "
              f"train: {t_parts}")
        print(f"{'':>{'30'}}  val:   {v_parts}")

        # ── checkpointing ────────────────────────────────────────────
        if (epoch + 1) % save_every == 0:
            _save_checkpoint(
                ckpt_dir / "last.pt", model, optimizer, scaler,
                epoch, best_val, cfg,
            )

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            _save_checkpoint(
                ckpt_dir / "best.pt", model, optimizer, scaler,
                epoch, best_val, cfg,
            )
            print(f"  ★ new best val loss: {best_val:.5f}")

    return best_val


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train HybridWarpNet")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stage", default="train", choices=["train", "finetune", "both"])
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mcfg = cfg["model"]
    model = HybridWarpNet(
        backbone=mcfg["backbone"],
        flow_channels=mcfg["flow_channels"],
        align_corners=mcfg["align_corners"],
        grid_sample_mode=mcfg["grid_sample_mode"],
        grid_sample_padding=mcfg["grid_sample_padding"],
    ).to(device)
    print(f"Model params: {count_parameters(model):,}")

    pairs, layout = discover_pairs(cfg["data"]["train_dir"])
    print(f"Data layout: {layout}  |  {len(pairs)} pairs found")

    if args.stage in ("train", "both"):
        train_stage(cfg, "train", model, pairs, device, resume_path=args.resume)

    if args.stage in ("finetune", "both"):
        resume = args.resume
        if args.stage == "both":
            resume = str(Path(cfg["checkpoint"]["dir"]) / "best.pt")
        elif resume is None:
            resume = str(Path(cfg["checkpoint"]["dir"]) / "best.pt")
        train_stage(
            cfg, "finetune", model, pairs, device,
            resume_path=resume, weights_only=True,
        )


if __name__ == "__main__":
    main()
