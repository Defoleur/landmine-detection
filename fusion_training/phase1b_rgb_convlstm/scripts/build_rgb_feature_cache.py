#!/usr/bin/env python3
"""Cache Phase 1A multiscale features for every temporal-manifest frame."""

import argparse
from pathlib import Path

from phase1b_rgb_convlstm.data.feature_cache import build_feature_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-root", type=Path)
    source.add_argument("--manifest", type=Path, help="Optional frame/temporal manifest source")
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    build_feature_cache(dataset_root=args.dataset_root, manifest_path=args.manifest,
                        encoder_checkpoint=args.encoder_checkpoint, output_dir=args.output_dir,
                        tile_size=args.tile_size, device=args.device)
    print(f"wrote feature cache to {args.output_dir}")


if __name__ == "__main__":
    main()
