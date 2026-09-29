#!/usr/bin/env python3
"""Validate cache provenance, tensors, shapes, dtype, and finite values."""

import argparse
from phase1b_rgb_convlstm.data.feature_cache import validate_feature_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir")
    args = parser.parse_args()
    report = validate_feature_cache(args.cache_dir)
    print(f"validated {report['records']} cached frames")


if __name__ == "__main__":
    main()
