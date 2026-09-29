"""Mask-aware multiscale RGB/rendered-thermal fusion model."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from phase1a_rgb_encoder.models import RGBEncoder
from phase1b_rgb_convlstm.rgb_temporal_encoder import RGBTemporalEncoder
from phase1c_thermal_image_encoder.models import ImageEncoder
from phase1a_rgb_encoder.models import RGBSegmentationDecoder


class MaskAwareFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels * 2 + 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, rgb, thermal, rgb_valid, thermal_valid):
        return self.block(
            torch.cat(
                (rgb * rgb_valid, thermal * thermal_valid, rgb_valid, thermal_valid),
                dim=1,
            )
        )


class MultiscaleFusion(nn.Module):
    def __init__(self):
        super().__init__()
        widths = {"1/2": 32, "1/4": 64, "1/8": 128}
        self.fusions = nn.ModuleDict(
            {scale: MaskAwareFusion(width) for scale, width in widths.items()}
        )

    def forward(self, rgb, thermal, rgb_valid, thermal_valid):
        return {
            scale: self.fusions[scale](
                rgb[scale], thermal[scale], rgb_valid[scale], thermal_valid[scale]
            )
            for scale in rgb
        }


class ImageFusionSegmenter(nn.Module):
    def __init__(
        self,
        rgb_encoder=None,
        thermal_encoder=None,
        rgb_temporal=None,
        thermal_temporal=None,
        decoder=None,
    ):
        super().__init__()
        self.rgb_encoder = rgb_encoder or RGBEncoder(pretrained=False)
        self.thermal_encoder = thermal_encoder or ImageEncoder(pretrained=False)
        self.rgb_temporal = rgb_temporal or RGBTemporalEncoder(
            use_condition_present=True
        )
        self.thermal_temporal = thermal_temporal or RGBTemporalEncoder(
            use_condition_present=True
        )
        self.fusion = MultiscaleFusion()
        self.decoder = decoder or RGBSegmentationDecoder()

    def _branch(self, sequence, encoder, temporal, availability, freeze):
        if sequence is None:
            return None, None
        batch, steps = sequence.shape[:2]
        frames = sequence.reshape(batch * steps, *sequence.shape[2:])
        with torch.no_grad() if freeze else torch.enable_grad():
            encoded = encoder(frames)
        features = {
            scale: value.reshape(batch, steps, *value.shape[1:])
            for scale, value in encoded.features.items()
        }
        output = temporal(features, condition_present=availability)
        if isinstance(output, tuple):
            output = output[0]
        # A branch is valid when at least one condition is available. Using
        # only the final slot masks valid earlier-only inputs during inference.
        branch_available = availability.any(dim=1)
        valid = {
            scale: branch_available
            .view(batch, 1, 1, 1)
            .to(value.dtype)
            .expand(-1, 1, value.shape[-2], value.shape[-1])
            for scale, value in output.items()
        }
        return output, valid

    def forward(
        self,
        rgb_sequence=None,
        thermal_sequence=None,
        rgb_availability=None,
        thermal_availability=None,
        freeze_branches=False,
    ):
        if rgb_sequence is None and thermal_sequence is None:
            raise ValueError("At least one modality must be available")
        source = rgb_sequence if rgb_sequence is not None else thermal_sequence
        batch, steps, _, height, width = source.shape
        device = source.device
        rgb_availability = (
            rgb_availability
            if rgb_availability is not None
            else torch.ones(batch, steps, device=device, dtype=torch.bool)
        )
        thermal_availability = (
            thermal_availability
            if thermal_availability is not None
            else torch.ones(batch, steps, device=device, dtype=torch.bool)
        )
        rgb, rv = self._branch(
            rgb_sequence,
            self.rgb_encoder,
            self.rgb_temporal,
            rgb_availability,
            freeze_branches,
        )
        thermal, tv = self._branch(
            thermal_sequence,
            self.thermal_encoder,
            self.thermal_temporal,
            thermal_availability,
            freeze_branches,
        )
        widths = {"1/2": 32, "1/4": 64, "1/8": 128}
        if rgb is None:
            rgb = {
                scale: source.new_zeros(
                    batch,
                    channels,
                    height // int(scale.split("/")[1]),
                    width // int(scale.split("/")[1]),
                )
                for scale, channels in widths.items()
            }
            rv = {scale: value[:, :1] for scale, value in rgb.items()}
        if thermal is None:
            thermal = {
                scale: source.new_zeros(value.shape) for scale, value in rgb.items()
            }
            tv = {scale: value[:, :1] for scale, value in thermal.items()}
        fused = self.fusion(rgb, thermal, rv, tv)
        return self.decoder(fused), {
            "features": fused,
            "rgb_valid": rv,
            "thermal_valid": tv,
        }
