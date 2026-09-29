#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import torch
from phase1d_thermal_image_convlstm.temporal_models import (
    ThermalImageTemporalSegmentationModel,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "phase1d_thermal_image_convlstm/artifacts/image_temporal_encoder.pt"
        ),
    )
    a = p.parse_args()
    model = ThermalImageTemporalSegmentationModel(use_condition_present=True)
    model.load_state_dict(
        torch.load(a.checkpoint, map_location="cpu", weights_only=False).get(
            "model_state_dict", {}
        ),
        strict=False,
    )
    meta = {
        "architecture": "ThermalImageTemporalSegmentationModel",
        "input": "[B,T,3,H,W]",
        "sequence_length": 12,
        "feature_contract": {"1/2": 32, "1/4": 64, "1/8": 128},
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.temporal.state_dict(), **meta}, a.output)
    a.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
