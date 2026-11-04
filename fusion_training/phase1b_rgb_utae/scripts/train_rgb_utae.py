#!/usr/bin/env python3
"""Train the RGB U-TAE model on the existing RGB sequence dataset contract."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from phase1a_rgb_encoder.losses import masked_bce_with_logits
from phase1a_rgb_encoder.metrics import SegmentationMetrics
from phase1a_rgb_encoder.models import RGBEncoder, RGBSegmentationDecoder
from phase1b_rgb_convlstm.data import (
    PointConditionDataset,
    make_virtual_samples,
    read_temporal_manifest,
    resample_virtual_samples,
    variable_sequence_collate,
)
from phase1b_rgb_utae.models import RGBUTAE
from phase1b_rgb_utae.time_encoding import ids_to_time_features


def load_encoder_checkpoint(encoder, checkpoint: Path) -> None:
    """Load an RGB encoder from standalone or full-model checkpoint formats."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    expected = encoder.state_dict()
    compatible = {}
    for prefix in ("", "encoder.", "rgb_encoder."):
        candidate = {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        compatible = {
            key: value
            for key, value in candidate.items()
            if key in expected and expected[key].shape == value.shape
        }
        if len(compatible) == len(expected):
            break
    missing = sorted(set(expected) - set(compatible))
    if missing:
        raise ValueError(f"{checkpoint} is missing compatible RGB encoder weights: {missing[:3]}")
    encoder.load_state_dict(compatible, strict=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader worker processes; use 0 to disable multiprocessing")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--virtual-samples-per-sequence", type=int, default=4)
    parser.add_argument("--virtual-validation-samples-per-sequence", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--amp", action="store_true", help="Enable CUDA automatic mixed precision")
    parser.add_argument("--amp-dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    parser.add_argument("--unfreeze-encoder", action="store_true",
                        help="Fine-tune the pretrained encoder instead of keeping it frozen")
    parser.add_argument("--use-single-input-mask", action="store_true",
                        help="Use the selected condition mask as target for one-input samples")
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    amp_enabled = args.amp and str(args.device).startswith("cuda")
    if amp_enabled and args.amp_dtype == "auto":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif args.amp_dtype == "bfloat16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    manifest_samples = read_temporal_manifest(args.manifest)
    samples = make_virtual_samples(
        [item for item in manifest_samples if item["split"] == "train"],
        args.virtual_samples_per_sequence,
    )
    validation_samples = make_virtual_samples(
        [item for item in manifest_samples if item["split"] == "val"],
        args.virtual_validation_samples_per_sequence,
        seed=10000,
    )
    if not samples or not validation_samples:
        raise SystemExit("manifest must contain both train and val samples")
    dataset = PointConditionDataset(samples, args.tile_size, augment=True,
                                    use_single_input_mask=args.use_single_input_mask)
    validation_dataset = PointConditionDataset(validation_samples, args.tile_size,
                                               use_single_input_mask=args.use_single_input_mask)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": str(args.device).startswith("cuda"),
        "prefetch_factor": 2,
    }
    if args.workers == 0:
        loader_options.pop("prefetch_factor")
    loader = DataLoader(dataset, shuffle=True, collate_fn=variable_sequence_collate, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False,
                                   collate_fn=variable_sequence_collate, **loader_options)
    freeze_encoder = not args.unfreeze_encoder
    encoder = RGBEncoder(pretrained=False)
    load_encoder_checkpoint(encoder, args.encoder_checkpoint)
    if freeze_encoder:
        encoder.requires_grad_(False)
    model = RGBUTAE(encoder, RGBSegmentationDecoder(), freeze_encoder=freeze_encoder).to(args.device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=3e-4,
        weight_decay=1e-4,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    start_epoch = 0
    if args.resume:
        payload = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(payload.get("model_state_dict", payload), strict=False)
        if payload.get("optimizer_state_dict"):
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_epoch = int(payload.get("epoch", 0))
        best_loss = float(payload.get("best_val_loss", payload.get("val_loss", best_loss)))
    metrics_mode = "a" if args.resume and (args.output_dir / "metrics.jsonl").exists() else "w"
    with (args.output_dir / "metrics.jsonl").open(metrics_mode) as metrics_file:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            if freeze_encoder:
                model.encoder.eval()
            dataset.samples = resample_virtual_samples(dataset.samples, epoch)
            total_loss = 0.0
            progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch")
            for batch in progress:
                optimizer.zero_grad(set_to_none=True)
                rgb = batch["rgb"].to(args.device)
                valid = batch["condition_present"].to(args.device)
                with (torch.autocast(device_type="cuda", dtype=amp_dtype)
                      if amp_enabled else nullcontext()):
                    logits = model(
                        rgb,
                        valid,
                        time_features=ids_to_time_features(
                            batch["season_ids"], batch["daytime_ids"]
                        ).to(args.device),
                    )
                    target = batch["target"].to(args.device)
                    target_valid = batch["target_validity"].to(args.device)
                    loss = masked_bce_with_logits(logits, target, target_valid, target_valid)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite training loss")
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                total_loss += float(loss.detach())
                progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
            epoch_loss = total_loss / max(len(loader), 1)
            model.eval()
            validation_loss = 0.0
            validation_metrics = SegmentationMetrics()
            with torch.inference_mode():
                for batch in tqdm(validation_loader, desc="Validation", unit="batch", leave=False):
                    with (torch.autocast(device_type="cuda", dtype=amp_dtype)
                          if amp_enabled else nullcontext()):
                        logits = model(
                            batch["rgb"].to(args.device), batch["condition_present"].to(args.device),
                            time_features=ids_to_time_features(batch["season_ids"], batch["daytime_ids"]).to(args.device),
                        )
                        validation_loss += float(masked_bce_with_logits(
                            logits, batch["target"].to(args.device), batch["target_validity"].to(args.device),
                            batch["target_validity"].to(args.device)
                        ))
                    validation_metrics.update(
                        logits,
                        batch["target"].to(args.device),
                        batch["target_validity"].to(args.device).bool(),
                        threshold=args.threshold,
                    )
            validation_loss /= max(len(validation_loader), 1)
            metric_values = {
                f"val_{key}": float(value)
                for key, value in validation_metrics.compute().items()
            }
            payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "architecture": "rgb_utae",
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "val_loss": validation_loss,
                "best_val_loss": min(best_loss, validation_loss),
                "tile_size": args.tile_size,
                "virtual_samples_per_sequence": args.virtual_samples_per_sequence,
                "virtual_validation_samples_per_sequence": args.virtual_validation_samples_per_sequence,
                "threshold": args.threshold,
                "time_encoding": "season_one_hot_4+daytime_one_hot_3",
                "amp": amp_enabled,
                "amp_dtype": str(amp_dtype).removeprefix("torch."),
                "workers": args.workers,
                "encoder_checkpoint": str(args.encoder_checkpoint.resolve()),
                "encoder_frozen": freeze_encoder,
                "use_single_input_mask": args.use_single_input_mask,
                **metric_values,
            }
            epoch_dir = args.output_dir / "epochs" / f"epoch_{epoch + 1:04d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            torch.save(payload, epoch_dir / "checkpoint.pt")
            torch.save(payload, args.output_dir / "last_checkpoint.pt")
            torch.save(payload, args.output_dir / "rgb_utae_checkpoint.pt")
            if validation_loss < best_loss:
                best_loss = validation_loss
                torch.save(payload, args.output_dir / "best_checkpoint.pt")
            record = {"epoch": epoch + 1, "train_loss": epoch_loss,
                      "val_loss": validation_loss, **metric_values}
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()
            print(json.dumps(record))


if __name__ == "__main__":
    main()
