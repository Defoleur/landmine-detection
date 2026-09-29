#!/usr/bin/env python3
"""Train mask-aware image fusion and decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from phase1a_rgb_encoder.losses import masked_bce_with_logits
from phase1a_rgb_encoder.metrics import SegmentationMetrics
from phase2_rgb_thermal_image_fusion.data import (
    ImageFusionSequenceDataset,
    read_manifest,
)
from phase2_rgb_thermal_image_fusion.models import ImageFusionSegmenter


def load_checkpoint(module, path, prefixes=()):
    if not path:
        return
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    if prefixes:
        state = {
            key.removeprefix(prefix): value
            for key, value in state.items()
            for prefix in prefixes
            if key.startswith(prefix)
        }
    expected = module.state_dict()
    state = {
        key: value
        for key, value in state.items()
        if key in expected and expected[key].shape == value.shape
    }
    module.load_state_dict(state, strict=False)


def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    metrics = SegmentationMetrics()
    with torch.inference_mode():
        for batch in loader:
            logits, _ = model(
                batch["rgb_sequence"].to(device),
                batch["thermal_image"].to(device),
                batch["rgb_available"].to(device),
                batch["thermal_available"].to(device),
                freeze_branches=True,
            )
            target = batch["target"].to(device)
            valid = batch["target_validity"].to(device)
            loss = masked_bce_with_logits(logits, target, valid, valid)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite image fusion validation loss")
            total += float(loss)
            metrics.update(logits, target, valid.bool())
    return total / max(len(loader), 1), metrics.compute()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "phase2_rgb_thermal_image_fusion/artifacts/image_fusion_manifest.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rgb-spatial-checkpoint", type=Path)
    parser.add_argument("--rgb-temporal-checkpoint", type=Path)
    parser.add_argument("--thermal-spatial-checkpoint", type=Path)
    parser.add_argument("--thermal-temporal-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    sequences = read_manifest(args.manifest)
    train = [item for item in sequences if item["split"] == "train"]
    val = [item for item in sequences if item["split"] == "val"]
    pin = args.device.startswith("cuda")
    train_loader = DataLoader(
        ImageFusionSequenceDataset(train, args.tile_size, True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        ImageFusionSequenceDataset(val, args.tile_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin,
    )
    model = ImageFusionSegmenter().to(args.device)
    load_checkpoint(
        model.rgb_encoder, args.rgb_spatial_checkpoint, ("encoder.", "rgb_encoder.")
    )
    load_checkpoint(
        model.rgb_temporal, args.rgb_temporal_checkpoint, ("temporal.", "rgb_temporal.")
    )
    load_checkpoint(
        model.thermal_encoder,
        args.thermal_spatial_checkpoint,
        ("encoder.", "thermal_encoder."),
    )
    load_checkpoint(
        model.thermal_temporal,
        args.thermal_temporal_checkpoint,
        ("temporal.", "thermal_temporal."),
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.fusion.parameters(), "lr": 3e-4},
            {"params": model.decoder.parameters(), "lr": 1e-4},
        ],
        weight_decay=1e-4,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    with (args.output_dir / "metrics.jsonl").open("w") as log:
        for epoch in range(args.epochs):
            model.train()
            total = 0.0
            progress = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch"
            )
            for batch in progress:
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(
                    batch["rgb_sequence"].to(args.device),
                    batch["thermal_image"].to(args.device),
                    batch["rgb_available"].to(args.device),
                    batch["thermal_available"].to(args.device),
                    freeze_branches=True,
                )
                loss = masked_bce_with_logits(
                    logits,
                    batch["target"].to(args.device),
                    batch["target_validity"].to(args.device),
                    batch["target_validity"].to(args.device),
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite image fusion loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach())
                progress.set_postfix(loss=f"{float(loss):.4f}")
            val_loss, val_metrics = evaluate(model, val_loader, args.device)
            record = {
                "epoch": epoch + 1,
                "train_loss": total / max(len(train_loader), 1),
                "val_loss": val_loss,
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            log.write(json.dumps(record) + "\n")
            payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "manifest": str(args.manifest.resolve()),
                "preprocessing": "ImageNet normalization for RGB and rendered thermal images",
            }
            torch.save(payload, args.output_dir / "last_image_fusion_checkpoint.pt")
            epoch_dir = args.output_dir / "epochs" / f"epoch_{epoch + 1:04d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            torch.save(payload, epoch_dir / "image_fusion_checkpoint.pt")
            if val_loss < best:
                best = val_loss
                torch.save(payload, args.output_dir / "best_image_fusion_checkpoint.pt")
            print(json.dumps(record))


if __name__ == "__main__":
    main()
