#!/usr/bin/env python3
"""Train the fixed 12-condition image temporal segmentation model."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from phase1a_rgb_encoder.losses import masked_bce_with_logits
from phase1a_rgb_encoder.metrics import SegmentationMetrics
from phase1d_thermal_image_convlstm.sequence import read_manifest
from phase1d_thermal_image_convlstm.temporal_data import ImageSequenceDataset
from phase1d_thermal_image_convlstm.temporal_models import (
    ThermalImageTemporalSegmentationModel,
)


def load_spatial(model, checkpoint, device):
    if checkpoint:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        state = payload.get("model_state_dict", payload)
        state = {
            k.removeprefix("encoder."): v
            for k, v in state.items()
            if k.startswith("encoder.")
        } or state
        model.encoder.load_state_dict(state, strict=False)


def evaluate(model, loader, device):
    model.eval()
    loss = 0.0
    metrics = SegmentationMetrics()
    with torch.inference_mode():
        for b in loader:
            logits, _ = model(
                b["image"].to(device),
                b["condition_present"].to(device),
                freeze_spatial=True,
            )
            target = b["target"].to(device)
            valid = b["target_validity"].to(device)
            batch_loss = masked_bce_with_logits(logits, target, valid, valid)
            if not torch.isfinite(batch_loss):
                raise FloatingPointError("Non-finite image temporal validation loss")
            loss += float(batch_loss)
            metrics.update(logits, target, valid.bool())
    return loss / max(len(loader), 1), metrics.compute()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--spatial-checkpoint", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--tile-size", type=int, default=640)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    seq = read_manifest(a.manifest)
    train = [x for x in seq if x["split"] == "train"]
    val = [x for x in seq if x["split"] == "val"]
    tl = DataLoader(
        ImageSequenceDataset(train, a.tile_size, True),
        batch_size=a.batch_size,
        shuffle=True,
        num_workers=a.workers,
        pin_memory=a.device.startswith("cuda"),
    )
    vl = DataLoader(
        ImageSequenceDataset(val, a.tile_size),
        batch_size=a.batch_size,
        shuffle=False,
        num_workers=a.workers,
        pin_memory=a.device.startswith("cuda"),
    )
    model = ThermalImageTemporalSegmentationModel(use_condition_present=True).to(
        a.device
    )
    load_spatial(model, a.spatial_checkpoint, a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    with (a.output_dir / "metrics.jsonl").open("w") as log:
        for epoch in range(a.epochs):
            model.train()
            total = 0.0
            progress = tqdm(
                tl,
                desc=f"Epoch {epoch + 1}/{a.epochs}",
                unit="batch",
            )
            for b in progress:
                opt.zero_grad(set_to_none=True)
                logits, _ = model(
                    b["image"].to(a.device), b["condition_present"].to(a.device)
                )
                loss = masked_bce_with_logits(
                    logits,
                    b["target"].to(a.device),
                    b["target_validity"].to(a.device),
                    torch.ones_like(b["target_validity"], device=a.device),
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite image temporal loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += float(loss.detach())
                progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
            vlss, met = evaluate(model, vl, a.device)
            rec = {
                "epoch": epoch + 1,
                "train_loss": total / max(len(tl), 1),
                "val_loss": vlss,
                **{f"val_{k}": v for k, v in met.items()},
            }
            log.write(json.dumps(rec) + "\n")
            payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "epoch": epoch + 1,
                "manifest": str(a.manifest.resolve()),
                "config": {
                    "batch_size": a.batch_size,
                    "tile_size": a.tile_size,
                    "workers": a.workers,
                    "use_condition_present": True,
                },
            }
            torch.save(payload, a.output_dir / "last_checkpoint.pt")
            epoch_dir = a.output_dir / "epochs" / f"epoch_{epoch + 1:04d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            torch.save(payload, epoch_dir / "checkpoint.pt")
            if vlss < best:
                best = vlss
                torch.save(payload, a.output_dir / "best_checkpoint.pt")
            print(json.dumps(rec))


if __name__ == "__main__":
    main()
