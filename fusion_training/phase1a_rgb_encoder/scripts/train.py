#!/usr/bin/env python3
"""Train the Phase 1A RGB segmentation model and export its encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

from phase1a_rgb_encoder.data import (RGBDataset, assign_spatial_splits,
                                      discover_records, read_manifest, write_manifest)
from phase1a_rgb_encoder.losses import masked_bce_with_logits
from phase1a_rgb_encoder.metrics import SegmentationMetrics
from phase1a_rgb_encoder.models import RGBSegmentationModel
from phase1a_rgb_encoder.training import export_encoder, save_checkpoint, seed_everything


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("phase1a_rgb_encoder/config.yaml"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("phase1a_rgb_encoder/artifacts/dataset_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("phase1a_rgb_encoder/artifacts/run_rgb_bce_v001"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download/use ImageNet ResNet-18 weights")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for validation metrics")
    return parser.parse_args()


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    total, count = 0.0, 0
    metrics = SegmentationMetrics()
    with torch.inference_mode():
        for batch in loader:
            logits, _ = model(batch["rgb"].to(device), batch["valid"].to(device), batch.get("timestamp"))
            loss = masked_bce_with_logits(logits, batch["target"].to(device),
                                          batch["target_valid"].to(device), batch["valid"].to(device))
            metrics.update(logits, batch["target"].to(device),
                           batch["target_valid"].to(device).bool() & batch["valid"].to(device).bool(), threshold)
            total += float(loss) * batch["rgb"].shape[0]
            count += batch["rgb"].shape[0]
    return total / max(count, 1), metrics.compute()


def main():
    args = arguments()
    config = yaml.safe_load(args.config.read_text()) if args.config.exists() else {}
    seed = int(config.get("experiment", {}).get("seed", 42))
    seed_everything(seed)
    dataset_root = args.dataset_root or Path(config["data"]["dataset_root"])
    if not dataset_root.is_absolute():
        dataset_root = (Path.cwd() / dataset_root).resolve()

    if args.manifest.exists():
        records = read_manifest(args.manifest)
    else:
        records = assign_spatial_splits(discover_records(dataset_root, config["data"].get("spatial_block_size", 10.0)), seed)
        write_manifest(records, args.manifest)
    train_records = [r for r in records if r["split"] == "train"]
    val_records = [r for r in records if r["split"] == "val"]
    tile_size = int(config.get("data", {}).get("tile_size", 256))
    train_loader = DataLoader(RGBDataset(train_records, augment=True, tile_size=tile_size), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=args.device.startswith("cuda"))
    val_loader = DataLoader(RGBDataset(val_records, augment=False, tile_size=tile_size), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=args.device.startswith("cuda"))

    model = RGBSegmentationModel(pretrained=not args.no_pretrained).to(args.device)
    training = config.get("training", {})
    new_lr = float(training.get("new_component_lr", 3e-4))
    pretrained_lr = float(training.get("pretrained_lr", 3e-5))
    optimizer = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": pretrained_lr},
                                   {"params": model.decoder.parameters(), "lr": new_lr}],
                                  weight_decay=float(training.get("weight_decay", 1e-4)))
    epochs = args.epochs or int(training.get("max_epochs", 100))
    warmup = int(training.get("warmup_epochs", 5))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    log_path = args.output_dir / "metrics.jsonl"
    with log_path.open("w") as logs:
        for epoch in range(epochs):
            if epoch == warmup:
                for parameter in model.encoder.parameters():
                    parameter.requires_grad = True
            model.train()
            train_total = 0.0
            progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
            for batch in progress:
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(batch["rgb"].to(args.device), batch["valid"].to(args.device), batch.get("timestamp"))
                loss = masked_bce_with_logits(logits, batch["target"].to(args.device),
                                              batch["target_valid"].to(args.device), batch["valid"].to(args.device))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                optimizer.step()
                train_total += float(loss.detach())
                progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
            val_loss, validation_metrics = evaluate(model, val_loader, args.device, args.threshold)
            record = {"epoch": epoch + 1, "train_loss": train_total / max(len(train_loader), 1), "val_loss": val_loss,
                      **{f"val_{key}": value for key, value in validation_metrics.items()}}
            logs.write(json.dumps(record) + "\n")
            logs.flush()
            save_checkpoint(args.output_dir / "last_checkpoint.pt", model, optimizer, epoch=epoch + 1,
                            best_metric=best, config=config, dataset_version=config.get("experiment", {}).get("dataset_version", ""),
                            split_version=config.get("experiment", {}).get("split_version", ""), seed=seed)
            epoch_dir = args.output_dir / "epochs" / f"epoch_{epoch + 1:04d}"
            save_checkpoint(epoch_dir / "checkpoint.pt", model, optimizer, epoch=epoch + 1,
                            best_metric=best, config=config, dataset_version=config.get("experiment", {}).get("dataset_version", ""),
                            split_version=config.get("experiment", {}).get("split_version", ""), seed=seed)
            torch.save(model.state_dict(), epoch_dir / "rgb_segmentation.pt")
            export_encoder(epoch_dir / "encoder_rgb.pt", model.encoder, config=config)
            if val_loss < best:
                best = val_loss
                save_checkpoint(args.output_dir / "best_checkpoint.pt", model, optimizer, epoch=epoch + 1,
                                best_metric=best, config=config, dataset_version=config.get("experiment", {}).get("dataset_version", ""),
                                split_version=config.get("experiment", {}).get("split_version", ""), seed=seed)
                torch.save(model.state_dict(), args.output_dir / "rgb_segmentation.pt")
                export_encoder(args.output_dir / "encoder_rgb.pt", model.encoder, config=config)
            print(f"epoch={epoch + 1}/{epochs} train_loss={record['train_loss']:.5f} val_loss={val_loss:.5f} "
                  f"val_dice={record['val_dice']:.4f} val_iou={record['val_iou']:.4f} "
                  f"val_precision={record['val_precision']:.4f} val_recall={record['val_recall']:.4f} "
                  f"val_pr_auc={record['val_pr_auc']:.4f}")


if __name__ == "__main__":
    main()
