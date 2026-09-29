#!/usr/bin/env python3
"""Train the 12-condition RGB temporal segmentation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

from phase1a_rgb_encoder.losses import masked_bce_with_logits
from phase1a_rgb_encoder.metrics import SegmentationMetrics
from phase1a_rgb_encoder.models import RGBEncoder
from phase1b_rgb_convlstm.data import CachedPointConditionDataset, PointConditionDataset, read_temporal_manifest
from phase1b_rgb_convlstm.models import CachedRGBTemporalSegmentationModel, RGBTemporalSegmentationModel


def load_phase1a(model, checkpoint, device):
    if not checkpoint:
        return
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("model_state_dict", payload)
    encoder_state = {key.removeprefix("encoder."): value for key, value in state.items() if key.startswith("encoder.")}
    decoder_state = {key.removeprefix("decoder."): value for key, value in state.items() if key.startswith("decoder.")}
    if not encoder_state:
        encoder_state = state
    if hasattr(model, "encoder"):
        model.encoder.load_state_dict(encoder_state, strict=False)
    if decoder_state:
        model.decoder.load_state_dict(decoder_state, strict=False)


def evaluate(model, loader, device, cached=False, threshold=0.5):
    model.eval()
    metrics = SegmentationMetrics()
    total = 0.0
    with torch.inference_mode():
        for batch in loader:
            if cached:
                features = {scale: value.to(device) for scale, value in batch["features"].items()}
                logits, _ = model(features, condition_present=batch["condition_present"].to(device))
            else:
                logits, _ = model(batch["rgb"].to(device), condition_present=batch["condition_present"].to(device),
                                  freeze_spatial=True)
            target = batch["target"].to(device)
            valid = batch["target_validity"].to(device)
            loss = masked_bce_with_logits(logits, target, valid, valid)
            total += float(loss)
            metrics.update(logits, target, valid.bool(), threshold=threshold)
    return total / max(len(loader), 1), metrics.compute()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase1a-checkpoint", type=Path)
    parser.add_argument("--feature-source", choices=("online", "cache"), default="online")
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("phase1b_rgb_convlstm/config.yaml"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    data_config, model_config, train_config = config["data"], config["model"], config["training"]
    samples = read_temporal_manifest(args.cache_manifest or args.manifest)
    cached = args.feature_source == "cache"
    if cached and not args.cache_manifest:
        raise SystemExit("--cache-manifest is required with --feature-source cache")
    train = [sample for sample in samples if sample["split"] == "train"]
    val = [sample for sample in samples if sample["split"] == "val"]
    batch_size = args.batch_size or int(data_config.get("batch_size", 1))
    workers = args.workers if args.workers is not None else int(data_config.get("workers", 0))
    dataset_type = CachedPointConditionDataset if cached else PointConditionDataset
    train_dataset = dataset_type(train, data_config.get("tile_size", 256)) if cached else dataset_type(train, data_config.get("tile_size", 256), augment=True)
    val_dataset = dataset_type(val, data_config.get("tile_size", 256)) if cached else dataset_type(val, data_config.get("tile_size", 256))
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size, shuffle=False, num_workers=workers)
    model_class = CachedRGBTemporalSegmentationModel if cached else RGBTemporalSegmentationModel
    model = model_class(use_condition_present=model_config.get("use_condition_present", False),
                        temporal_mean=model_config.get("temporal_mean", False)).to(args.device)
    load_phase1a(model, args.phase1a_checkpoint, args.device)
    stage = train_config.get("stage", "temporal_only")
    if stage == "temporal_only":
        frozen = ((*model.encoder.parameters(),) if hasattr(model, "encoder") else ()) + (*model.decoder.parameters(),)
        for parameter in frozen:
            parameter.requires_grad = False
    elif stage == "decoder_adaptation":
        if hasattr(model, "encoder"):
            for parameter in model.encoder.parameters():
                parameter.requires_grad = False
    groups = [{"params": [p for p in model.temporal.parameters() if p.requires_grad], "lr": float(train_config.get("temporal_lr", 3e-4))}]
    if any(p.requires_grad for p in model.decoder.parameters()):
        groups.append({"params": [p for p in model.decoder.parameters() if p.requires_grad], "lr": float(train_config.get("decoder_lr", 1e-4))})
    if hasattr(model, "encoder") and any(p.requires_grad for p in model.encoder.parameters()):
        groups.append({"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": float(train_config.get("rgb_encoder_lr", 1e-5))})
    optimizer = torch.optim.AdamW(groups, weight_decay=float(train_config.get("weight_decay", 1e-4)))
    epochs = args.epochs or int(train_config.get("epochs", 50))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    with (args.output_dir / "metrics.jsonl").open("w") as logs:
        for epoch in range(epochs):
            model.train()
            train_total = 0.0
            progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
            for batch in progress:
                optimizer.zero_grad(set_to_none=True)
                if cached:
                    features = {scale: value.to(args.device) for scale, value in batch["features"].items()}
                    logits, _ = model(features, condition_present=batch["condition_present"].to(args.device))
                else:
                    logits, _ = model(batch["rgb"].to(args.device), condition_present=batch["condition_present"].to(args.device),
                                      freeze_spatial=not any(p.requires_grad for p in model.encoder.parameters()))
                valid = batch["target_validity"].to(args.device)
                loss = masked_bce_with_logits(logits, batch["target"].to(args.device), valid, valid)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config.get("max_grad_norm", 1.0)))
                optimizer.step()
                train_total += float(loss.detach())
                progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
            val_loss, metrics = evaluate(model, val_loader, args.device, cached=cached)
            record = {"epoch": epoch + 1, "train_loss": train_total / max(len(train_loader), 1),
                      "val_loss": val_loss, **{f"val_{k}": v for k, v in metrics.items()}}
            logs.write(json.dumps(record) + "\n")
            logs.flush()
            payload = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                       "epoch": epoch + 1, "best_metric": best, "config": config,
                       "manifest": str(args.manifest.resolve())}
            torch.save(payload, args.output_dir / "last_checkpoint.pt")
            epoch_dir = args.output_dir / "epochs" / f"epoch_{epoch + 1:04d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            torch.save(payload, epoch_dir / "checkpoint.pt")
            torch.save(model.state_dict(), epoch_dir / "rgb_temporal_segmentation.pt")
            if val_loss < best:
                best = val_loss
                payload["best_metric"] = best
                torch.save(payload, args.output_dir / "best_checkpoint.pt")
                torch.save(model.state_dict(), args.output_dir / "rgb_temporal_segmentation.pt")
            print(json.dumps(record))


if __name__ == "__main__":
    main()
