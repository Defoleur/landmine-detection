#!/usr/bin/env python3
"""Train asynchronous RGB/thermal U-TAE fusion."""

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
from phase2_rgb_thermal_image_fusion.data import (
    ImageFusionSequenceDataset,
    make_virtual_sequences,
    read_manifest,
    resample_virtual_sequences,
    variable_sequence_collate,
)
from phase2_rgb_thermal_image_utae.manifest import id_time_features, legacy_modality_ages
from phase2_rgb_thermal_image_utae.models import AsyncUTAESegmenter


def load_module_checkpoint(module, checkpoint: Path, prefix: str = "") -> None:
    """Load a module from either a standalone or a full-model checkpoint."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {
        key.removeprefix("module."): value
        for key, value in state.items()
    }
    expected = module.state_dict()
    prefixes = [prefix] if prefix else ["", "encoder.", "rgb_encoder.", "thermal_encoder."]
    compatible = {}
    for candidate_prefix in prefixes:
        candidate = {
            key.removeprefix(candidate_prefix): value
            for key, value in state.items()
            if key.startswith(candidate_prefix)
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
        raise ValueError(
            f"{checkpoint} does not contain a complete compatible {module.__class__.__name__} "
            f"checkpoint; missing {missing[:3]}"
        )
    module.load_state_dict(compatible, strict=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rgb-encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-utae-checkpoint", type=Path, required=True)
    parser.add_argument("--thermal-encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--thermal-utae-checkpoint", type=Path, required=True)
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
    manifest_samples = read_manifest(args.manifest)
    samples = make_virtual_sequences(
        [item for item in manifest_samples if item["split"] == "train"],
        args.virtual_samples_per_sequence,
        use_single_input_mask=args.use_single_input_mask,
    )
    validation_samples = make_virtual_sequences(
        [item for item in manifest_samples if item["split"] == "val"],
        args.virtual_validation_samples_per_sequence,
        seed=10000,
        use_single_input_mask=args.use_single_input_mask,
    )
    if not samples or not validation_samples:
        raise SystemExit("manifest must contain both train and val samples")
    dataset = ImageFusionSequenceDataset(
        samples, args.tile_size, augment=True,
        use_single_input_mask=args.use_single_input_mask,
    )
    validation_dataset = ImageFusionSequenceDataset(
        validation_samples, args.tile_size,
        use_single_input_mask=args.use_single_input_mask,
    )
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
    model = AsyncUTAESegmenter().to(args.device)
    load_module_checkpoint(model.rgb_encoder, args.rgb_encoder_checkpoint)
    load_module_checkpoint(model.rgb_temporal, args.rgb_utae_checkpoint, "temporal.")
    load_module_checkpoint(model.thermal_encoder, args.thermal_encoder_checkpoint)
    load_module_checkpoint(model.thermal_temporal, args.thermal_utae_checkpoint, "temporal.")
    model.freeze_pretrained_branches()
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
            model.rgb_encoder.eval()
            model.thermal_encoder.eval()
            model.rgb_temporal.eval()
            model.thermal_temporal.eval()
            dataset.sequences = resample_virtual_sequences(
                dataset.sequences, epoch,
                use_single_input_mask=args.use_single_input_mask,
            )
            total_loss = 0.0
            progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch")
            for batch in progress:
                optimizer.zero_grad(set_to_none=True)
                rgb_ages, thermal_ages = legacy_modality_ages(
                    batch["season_ids"], batch["daytime_ids"], batch["rgb_available"], batch["thermal_available"]
                )
                with (torch.autocast(device_type="cuda", dtype=amp_dtype)
                      if amp_enabled else nullcontext()):
                    logits = model(
                        batch["rgb_sequence"].to(args.device), batch["thermal_image"].to(args.device),
                        batch["rgb_available"].to(args.device), batch["thermal_available"].to(args.device),
                        rgb_time_features=id_time_features(batch["season_ids"], batch["daytime_ids"]).to(args.device),
                        thermal_time_features=id_time_features(batch["season_ids"], batch["daytime_ids"]).to(args.device),
                        rgb_ages=rgb_ages.to(args.device), thermal_ages=thermal_ages.to(args.device),
                    )
                    target = batch["target"].to(args.device)
                    valid = batch["target_validity"].to(args.device)
                    loss = masked_bce_with_logits(logits, target, valid, valid)
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
                    rgb_ages, thermal_ages = legacy_modality_ages(
                        batch["season_ids"], batch["daytime_ids"],
                        batch["rgb_available"], batch["thermal_available"]
                    )
                    time_features = id_time_features(batch["season_ids"], batch["daytime_ids"]).to(args.device)
                    with (torch.autocast(device_type="cuda", dtype=amp_dtype)
                          if amp_enabled else nullcontext()):
                        logits = model(
                            batch["rgb_sequence"].to(args.device), batch["thermal_image"].to(args.device),
                            batch["rgb_available"].to(args.device), batch["thermal_available"].to(args.device),
                            rgb_time_features=time_features, thermal_time_features=time_features,
                            rgb_ages=rgb_ages.to(args.device), thermal_ages=thermal_ages.to(args.device),
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
            payload = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                       "architecture": "async_utae_fusion", "epoch": epoch + 1,
                       "train_loss": epoch_loss, "val_loss": validation_loss,
                       "best_val_loss": min(best_loss, validation_loss), **metric_values}
            payload.update({
                "tile_size": args.tile_size,
                "virtual_samples_per_sequence": args.virtual_samples_per_sequence,
                "virtual_validation_samples_per_sequence": args.virtual_validation_samples_per_sequence,
                "threshold": args.threshold,
                "time_encoding": "season_one_hot_4+daytime_one_hot_3",
                "amp": amp_enabled,
                "amp_dtype": str(amp_dtype).removeprefix("torch."),
                "workers": args.workers,
                "frozen_pretrained_branches": True,
                "rgb_encoder_checkpoint": str(args.rgb_encoder_checkpoint.resolve()),
                "rgb_utae_checkpoint": str(args.rgb_utae_checkpoint.resolve()),
                "thermal_encoder_checkpoint": str(args.thermal_encoder_checkpoint.resolve()),
                "thermal_utae_checkpoint": str(args.thermal_utae_checkpoint.resolve()),
                "use_single_input_mask": args.use_single_input_mask,
            })
            epoch_dir = args.output_dir / "epochs" / f"epoch_{epoch + 1:04d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            torch.save(payload, epoch_dir / "checkpoint.pt")
            torch.save(payload, args.output_dir / "last_checkpoint.pt")
            torch.save(payload, args.output_dir / "fusion_utae_checkpoint.pt")
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
