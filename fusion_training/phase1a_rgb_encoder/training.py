"""Small, explicit training helpers for Phase 1A experiments."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from .losses import masked_bce_with_logits


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_step(model, batch, optimizer, device=None):
    device = device or next(model.parameters()).device
    model.train()
    rgb = batch["rgb"].to(device)
    valid = batch["valid"].to(device)
    logits, _ = model(rgb, valid, batch.get("timestamp"))
    target = batch["target"].to(device)
    target_valid = batch["target_valid"].to(device)
    loss = masked_bce_with_logits(logits, target, target_valid, valid)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite masked BCE loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=0, global_step=0,
                    best_metric=None, config=None, dataset_version="", split_version="", seed=42):
    payload = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
               "scheduler_state_dict": scheduler.state_dict() if scheduler else None, "epoch": epoch,
               "global_step": global_step, "best_metric": best_metric, "config": config,
               "dataset_version": dataset_version, "split_version": split_version, "random_seed": seed}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def export_encoder(path, encoder, config=None, preprocessing=None):
    """Write encoder-only state and sidecar contracts, independent of decoder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": encoder.state_dict(), "config": config or {}}, path)
    (path.parent / "encoder_rgb_preprocessing.json").write_text(json.dumps(preprocessing or {
        "scale": "uint8 / 255", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}, indent=2))
    (path.parent / "encoder_rgb_feature_contract.json").write_text(json.dumps({
        "input_channels": 4, "scales": ["1/2", "1/4", "1/8"], "channels": [32, 64, 128]}, indent=2))


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
