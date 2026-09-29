#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import torch
from matplotlib import colormaps
from PIL import Image
from phase1d_thermal_image_convlstm.sequence import read_manifest
from phase1d_thermal_image_convlstm.temporal_data import ImageSequenceDataset
from phase1d_thermal_image_convlstm.temporal_models import (
    ThermalImageTemporalSegmentationModel,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--sequence-id", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    sample = next(
        x for x in read_manifest(a.manifest) if x["sequence_id"] == a.sequence_id
    )
    b = ImageSequenceDataset([sample], a.tile_size)[0]
    model = (
        ThermalImageTemporalSegmentationModel(use_condition_present=True)
        .to(a.device)
        .eval()
    )
    model.load_state_dict(
        torch.load(a.checkpoint, map_location=a.device, weights_only=False).get(
            "model_state_dict", {}
        ),
        strict=False,
    )
    with torch.inference_mode():
        logits, _ = model(
            b["image"].unsqueeze(0).to(a.device),
            b["condition_present"].unsqueeze(0).to(a.device),
            freeze_spatial=True,
        )
    prob = torch.sigmoid(logits[0, 0]).cpu().numpy()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((prob >= a.threshold).astype("uint8") * 255).save(a.output)
    Image.fromarray((colormaps["inferno"](prob)[..., :3] * 255).astype("uint8")).save(
        a.output.with_name(a.output.stem + "_probability" + a.output.suffix)
    )


if __name__ == "__main__":
    main()
