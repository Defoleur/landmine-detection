"""Three-channel thermal-image U-TAE model."""

from __future__ import annotations

import torch
from torch import nn

from phase1b_rgb_utae.models import UTAETemporalAggregator
from phase1c_thermal_image_encoder.models import ImageEncoder, ImageSegmentationDecoder


class ThermalImageUTAE(nn.Module):
    """Thermal image encoder and U-TAE temporal aggregation."""

    def __init__(self, spatial=None, temporal=None, decoder=None, freeze_spatial=False):
        super().__init__()
        self.encoder = spatial or ImageEncoder(input_channels=3)
        self.temporal = temporal or UTAETemporalAggregator()
        self.decoder = decoder or ImageSegmentationDecoder()
        self.freeze_spatial = freeze_spatial

    def forward(self, image, frame_valid, time_features=None, feature_valid=None, return_details=False):
        if image.ndim != 5 or image.shape[2] != 3:
            raise ValueError("image must have shape [B,T,3,H,W]")
        batch, steps = image.shape[:2]
        frames = image.reshape(batch * steps, *image.shape[2:])
        with torch.no_grad() if self.freeze_spatial else torch.enable_grad():
            encoded = self.encoder(frames)
        features = {
            int(scale.split("/")[-1]): value.reshape(batch, steps, *value.shape[1:])
            for scale, value in encoded.features.items()
        }
        if feature_valid is None:
            feature_valid = {
                scale: frame_valid[:, :, None, None, None].expand(
                    batch, steps, 1, value.shape[-2], value.shape[-1]
                )
                for scale, value in features.items()
            }
        aggregate_result = self.temporal(
            features, frame_valid, feature_valid, time_features, return_attention=return_details
        )
        if return_details:
            aggregate, availability, temporal_details = aggregate_result
        else:
            aggregate, availability = aggregate_result
        logits = self.decoder({f"1/{scale}": value for scale, value in aggregate.items()})
        if return_details:
            return logits, {"features": aggregate, "availability": availability, **temporal_details}
        return logits
