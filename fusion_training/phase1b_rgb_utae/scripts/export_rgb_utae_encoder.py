#!/usr/bin/env python3
"""Export the RGB U-TAE temporal module and architecture metadata."""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    temporal = {key.removeprefix("temporal."): value for key, value in state.items()
                if key.startswith("temporal.")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": temporal, "architecture": "utae_temporal_aggregator",
                "temporal_features": 7}, args.output)
    print(json.dumps({"output": str(args.output), "parameters": len(temporal)}, indent=2))


if __name__ == "__main__":
    main()
