#!/usr/bin/env python3
"""Visualize RGB U-TAE attention from a legacy RGB sequence sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from phase1a_rgb_encoder.models import RGBEncoder, RGBSegmentationDecoder
from phase1b_rgb_convlstm.data import PointConditionDataset, read_temporal_manifest
from phase1b_rgb_utae.models import RGBUTAE
from phase1b_rgb_utae.time_encoding import ids_to_time_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=640)
    args = parser.parse_args()
    sample = next(item for item in read_temporal_manifest(args.manifest) if item["sequence_id"] == args.sequence_id)
    item = PointConditionDataset([sample], args.tile_size)[0]
    model = RGBUTAE(RGBEncoder(), RGBSegmentationDecoder()).eval()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload), strict=False)
    with torch.inference_mode():
        _, details = model(item["rgb"].unsqueeze(0), item["condition_present"].unsqueeze(0),
                           ids_to_time_features(item["season_ids"].unsqueeze(0), item["daytime_ids"].unsqueeze(0)),
                           return_details=True)
    attention = details["attention"][0].mean(dim=0)
    figure, axes = plt.subplots(1, attention.shape[0], figsize=(3 * attention.shape[0], 3), squeeze=False)
    for index, axis in enumerate(axes[0]):
        axis.imshow(attention[index], cmap="magma", vmin=0, vmax=float(attention.max()))
        axis.set_title(f"t={index + 1}")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
