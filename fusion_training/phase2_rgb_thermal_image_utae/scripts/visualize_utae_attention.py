#!/usr/bin/env python3
"""Save RGB/thermal U-TAE attention maps with one shared color scale."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from phase2_rgb_thermal_image_fusion.data import ImageFusionSequenceDataset, read_manifest
from phase2_rgb_thermal_image_utae.manifest import id_time_features, legacy_modality_ages
from phase2_rgb_thermal_image_utae.models import AsyncUTAESegmenter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    sequence = next(item for item in read_manifest(args.manifest) if item["sequence_id"] == args.sequence_id)
    item = ImageFusionSequenceDataset([sequence], args.tile_size)[0]
    model = AsyncUTAESegmenter().to(args.device).eval()
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload), strict=False)
    time_features = id_time_features(item["season_ids"].unsqueeze(0), item["daytime_ids"].unsqueeze(0)).to(args.device)
    rgb_ages, thermal_ages = legacy_modality_ages(
        item["season_ids"].unsqueeze(0), item["daytime_ids"].unsqueeze(0),
        item["rgb_available"].unsqueeze(0), item["thermal_available"].unsqueeze(0),
    )
    with torch.inference_mode():
        _, details = model(
            item["rgb_sequence"].unsqueeze(0).to(args.device), item["thermal_image"].unsqueeze(0).to(args.device),
            item["rgb_available"].unsqueeze(0).to(args.device), item["thermal_available"].unsqueeze(0).to(args.device),
            rgb_time_features=time_features, thermal_time_features=time_features, return_details=True,
            rgb_ages=rgb_ages.to(args.device), thermal_ages=thermal_ages.to(args.device),
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for modality, branch in details.items():
        if modality not in ("rgb", "thermal"):
            continue
        attention = branch["attention"][0].detach().cpu()
        for head in range(attention.shape[0]):
            upper = float(torch.quantile(attention[head], 0.99))
            figure, axes = plt.subplots(1, attention.shape[1], figsize=(3 * attention.shape[1], 3), squeeze=False)
            for index, axis in enumerate(axes[0]):
                axis.imshow((attention[head, index] / max(upper, 1e-8)).clamp(0, 1), cmap="magma", vmin=0, vmax=1)
                axis.set_title(f"{modality} head {head} t={index + 1}")
                axis.set_xticks([])
                axis.set_yticks([])
            figure.tight_layout()
            figure.savefig(args.output_dir / f"{modality}_head_{head}.png", dpi=160)
            plt.close(figure)


if __name__ == "__main__":
    main()
