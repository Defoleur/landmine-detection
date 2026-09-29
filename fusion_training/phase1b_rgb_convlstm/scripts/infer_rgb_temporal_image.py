#!/usr/bin/env python3
"""Run the RGB ConvLSTM on one image as a one-step sequence."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import torch
import yaml
from PIL import Image

from phase1a_rgb_encoder.models import RGBSegmentationModel
from phase1a_rgb_encoder.scripts.visualize_image import load_image
from phase1b_rgb_convlstm.models import RGBTemporalSegmentationModel


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase1a-checkpoint", type=Path,
                        help="Optional RGB encoder+decoder checkpoint for comparison")
    parser.add_argument("--config", type=Path, default=Path("phase1b_rgb_convlstm/config.yaml"))
    parser.add_argument("--output", type=Path, help="Output thresholded mask PNG")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("phase1b_rgb_convlstm/artifacts/single_image_outputs"),
                        help="Directory for default image-named outputs")
    parser.add_argument("--probability-output", type=Path,
                        help="Output grayscale probability PNG; defaults to <output>_probability.png")
    parser.add_argument("--crop-square", action="store_true",
                        help="Center-crop the input image to a square before resizing")
    parser.add_argument("--repeat-image", type=int, choices=range(2, 13), metavar="COUNT",
                        help="Repeat the same image across COUNT temporal conditions (2-12)")
    parser.add_argument("--lstm-threshold", type=float, default=0.5)
    parser.add_argument("--phase1a-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def save_visualization(image, probability, prediction, output, lstm_threshold,
                       phase1a=None, phase1a_threshold=0.5):
    panels = [(image, "RGB input", None),
              (probability, "probability", "inferno"),
              (prediction, f"prediction >= {lstm_threshold:g}", "Greens")]
    if phase1a is not None:
        phase1a_probability, phase1a_prediction = phase1a
        panels.extend([(phase1a_probability, "baseline probability", "inferno"),
                       (phase1a_prediction, f"baseline >= {phase1a_threshold:g}", "Greens")])
    figure, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    axes = np.atleast_1d(axes)
    for axis, (panel, title, cmap) in zip(axes, panels):
        if cmap is None:
            axis.imshow(panel)
        elif title.endswith("probability"):
            axis.imshow(panel, cmap=cmap, vmin=0, vmax=1)
        else:
            axis.imshow(image)
            axis.imshow(panel, cmap=cmap, alpha=0.45, vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle("RGB temporal single-image inference")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def save_encoder_visualization(image, features, output):
    panels = []
    for scale in ("1/2", "1/4", "1/8"):
        activation = features[scale][0].detach().float().cpu().pow(2).mean(dim=0).sqrt().numpy()
        maximum = float(activation.max())
        if maximum > 0:
            activation /= maximum
        panels.append((activation, f"LSTM output {scale}"))

    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    for axis, (panel, title) in zip(axes[:3], panels):
        axis.imshow(panel, cmap="inferno", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    axes[3].imshow(image)
    axes[3].set_title("RGB input")
    axes[3].axis("off")
    figure.suptitle("ConvLSTM encoder features before decoder")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main():
    args = arguments()
    if not 0 <= args.lstm_threshold <= 1 or not 0 <= args.phase1a_threshold <= 1:
        raise SystemExit("--lstm-threshold and --phase1a-threshold must be between 0 and 1")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    config = yaml.safe_load(args.config.read_text())
    tile_size = int(config["data"].get("tile_size", 256))
    model_config = config.get("model", {})
    image_output_dir = args.output_dir / args.image.stem
    if args.output is None:
        args.output = image_output_dir / f"{args.image.stem}_lstm_mask.png"
    probability_output = args.probability_output or args.output.with_name(
        f"{args.image.stem}_lstm_probability{args.output.suffix}")
    visualization_output = args.output.with_name(f"{args.image.stem}_comparison.png")
    encoder_output = args.output.with_name(f"{args.image.stem}_lstm_encoder.png")

    image, rgb, _, _ = load_image(args.image, tile_size, args.crop_square)
    rgb = rgb.to(args.device).unsqueeze(1)
    if args.repeat_image_12:
        rgb = rgb.repeat(1, args.repeat_image_12, 1, 1, 1)
    steps = rgb.shape[1]
    validity = torch.ones((1, steps, 1, tile_size, tile_size), device=args.device, dtype=rgb.dtype)
    present = torch.ones((1, steps), device=args.device, dtype=torch.bool)

    model = RGBTemporalSegmentationModel(
        use_condition_present=model_config.get("use_condition_present", False),
        temporal_mean=model_config.get("temporal_mean", False),
    ).to(args.device).eval()
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload), strict=False)

    with torch.inference_mode():
        logits, temporal_details = model(rgb, rgb_validity_sequence=validity, condition_present=present)
    probability = torch.sigmoid(logits[0, 0]).cpu().numpy()
    prediction = probability >= args.lstm_threshold
    args.output.parent.mkdir(parents=True, exist_ok=True)
    probability_output.parent.mkdir(parents=True, exist_ok=True)
    save_encoder_visualization(image, temporal_details["features"], encoder_output)

    phase1a_result = None
    if args.phase1a_checkpoint:
        if not args.phase1a_checkpoint.exists():
            raise FileNotFoundError(f"Phase 1A checkpoint not found: {args.phase1a_checkpoint}")
        phase1a_model = RGBSegmentationModel(pretrained=False).to(args.device).eval()
        phase1a_payload = torch.load(args.phase1a_checkpoint, map_location=args.device, weights_only=False)
        phase1a_model.load_state_dict(phase1a_payload.get("model_state_dict", phase1a_payload))
        with torch.inference_mode():
            phase1a_logits, _ = phase1a_model(rgb[:, 0], validity[:, 0])
        phase1a_probability = torch.sigmoid(phase1a_logits[0, 0]).cpu().numpy()
        phase1a_prediction = phase1a_probability >= args.phase1a_threshold
        phase1a_result = (phase1a_probability, phase1a_prediction)
        phase1a_output = args.output.with_name(f"{args.image.stem}_phase1a_mask{args.output.suffix}")
        phase1a_probability_output = probability_output.with_name(
            f"{args.image.stem}_phase1a_probability{probability_output.suffix}")
        Image.fromarray((phase1a_prediction.astype(np.uint8) * 255)).save(phase1a_output)
        Image.fromarray(
            (colormaps["inferno"](phase1a_probability.clip(0, 1))[..., :3] * 255).astype(
                np.uint8
            )
        ).save(phase1a_probability_output)
        print(f"wrote Phase 1A mask: {phase1a_output}")
        print(f"wrote Phase 1A probability: {phase1a_probability_output}")
        print(f"Phase 1A probability min={phase1a_probability.min():.6f} "
              f"max={phase1a_probability.max():.6f} mean={phase1a_probability.mean():.6f} "
              f"positive_pixels={int(phase1a_prediction.sum())}")

    Image.fromarray((prediction.astype(np.uint8) * 255)).save(args.output)
    Image.fromarray(
        (colormaps["inferno"](probability.clip(0, 1))[..., :3] * 255).astype(np.uint8)
    ).save(probability_output)
    save_visualization(image, probability, prediction, visualization_output, args.lstm_threshold,
                       phase1a_result, args.phase1a_threshold)

    print(f"wrote mask: {args.output}")
    print(f"wrote probability: {probability_output}")
    print(f"wrote LSTM encoder visualization: {encoder_output}")
    print(f"wrote visualization: {visualization_output}")
    print(f"probability min={probability.min():.6f} max={probability.max():.6f} "
          f"mean={probability.mean():.6f} positive_pixels={int(prediction.sum())}")


if __name__ == "__main__":
    main()
