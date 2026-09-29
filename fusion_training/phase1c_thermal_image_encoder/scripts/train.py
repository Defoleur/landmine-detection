#!/usr/bin/env python3
"""Train the image-based Phase 1C segmentation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from phase1a_rgb_encoder.metrics import SegmentationMetrics
from phase1c_thermal_image_encoder.data import ImageSegmentationDataset, read_manifest
from phase1c_thermal_image_encoder.models import ImageSegmentationModel


def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    metrics = SegmentationMetrics()
    with torch.inference_mode():
        for batch in loader:
            logits, _ = model(batch["image"].to(device))
            target = batch["target"].to(device)
            total += float(torch.nn.functional.binary_cross_entropy_with_logits(logits, target))
            metrics.update(logits, target, batch["target_validity"].to(device).bool())
    return total / max(len(loader), 1), metrics.compute()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("phase1c_thermal_image_encoder/config.yaml"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("phase1c_thermal_image_encoder/artifacts/run_image"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    data_config, train_config = config["data"], config["training"]
    manifest = args.manifest or Path(data_config["manifest"])
    records = read_manifest(manifest)
    tile_size = int(data_config.get("tile_size", 640))
    batch_size = args.batch_size or int(train_config.get("batch_size", 8))
    workers = int(train_config.get("num_workers", 0))
    train_loader = DataLoader(ImageSegmentationDataset([r for r in records if r["split"] == "train"], tile_size, True),
                              batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(ImageSegmentationDataset([r for r in records if r["split"] == "val"], tile_size),
                            batch_size=batch_size, shuffle=False, num_workers=workers)
    model = ImageSegmentationModel(bool(config["model"].get("pretrained", False))).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_config.get("learning_rate", 3e-4)),
                                  weight_decay=float(train_config.get("weight_decay", 1e-4)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    epochs = args.epochs or int(train_config.get("epochs", 50))
    with (args.output_dir / "metrics.jsonl").open("w") as log:
        for epoch in range(epochs):
            model.train()
            total = 0.0
            progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
            for batch in progress:
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(batch["image"].to(args.device))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["target"].to(args.device))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite image segmentation loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config.get("max_grad_norm", 1.0)))
                optimizer.step()
                total += float(loss.detach())
                progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
            val_loss, metrics = evaluate(model, val_loader, args.device)
            record = {"epoch": epoch + 1, "train_loss": total / max(len(train_loader), 1), "val_loss": val_loss,
                      **{f"val_{key}": value for key, value in metrics.items()}}
            log.write(json.dumps(record) + "\n")
            payload = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                       "epoch": epoch + 1, "config": config, "manifest": str(manifest.resolve())}
            torch.save(payload, args.output_dir / "last_checkpoint.pt")
            epoch_dir = args.output_dir / "epochs" / f"epoch_{epoch + 1:04d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            torch.save(payload, epoch_dir / "checkpoint.pt")
            if val_loss < best:
                best = val_loss
                torch.save(payload, args.output_dir / "best_checkpoint.pt")
            print(json.dumps(record))


if __name__ == "__main__":
    main()
