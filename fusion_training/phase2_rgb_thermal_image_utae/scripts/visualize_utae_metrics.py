#!/usr/bin/env python3
"""Plot metric values from a U-TAE fusion test result JSON."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--metric", default="dice")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    scenarios = data.get("scenarios", data)
    labels = [item.get("id", str(index)) for index, item in enumerate(scenarios)]
    values = [item.get("metrics", item).get(args.metric, float("nan")) for item in scenarios]
    figure, axis = plt.subplots(figsize=(max(10, len(labels) * 0.35), 5))
    axis.plot(range(len(values)), values, marker=".")
    axis.set_xticks(range(len(labels)), labels, rotation=90)
    axis.set_ylabel(args.metric)
    axis.set_title(f"U-TAE fusion {args.metric}")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
