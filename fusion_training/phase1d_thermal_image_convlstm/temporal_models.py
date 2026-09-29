"""Image encoder + shared Phase 1B ConvLSTM + image decoder."""

import torch
from torch import nn

from phase1b_rgb_convlstm.rgb_temporal_encoder import RGBTemporalEncoder
from phase1c_thermal_image_encoder.models import ImageEncoder, ImageSegmentationDecoder


class ThermalImageTemporalSegmentationModel(nn.Module):
    def __init__(
        self, spatial=None, temporal=None, decoder=None, use_condition_present=True
    ):
        super().__init__()
        self.encoder = spatial or ImageEncoder()
        self.temporal = temporal or RGBTemporalEncoder(
            use_condition_present=use_condition_present
        )
        self.decoder = decoder or ImageSegmentationDecoder()

    def forward(self, image_sequence, condition_present=None, freeze_spatial=False):
        if image_sequence.ndim != 5 or image_sequence.shape[2] != 3:
            raise ValueError("image_sequence must have shape [B,T,3,H,W]")
        batch, steps = image_sequence.shape[:2]
        frames = image_sequence.reshape(batch * steps, 3, *image_sequence.shape[-2:])
        context = torch.no_grad() if freeze_spatial else torch.enable_grad()
        with context:
            encoded = self.encoder(frames)
        features = {
            scale: value.reshape(batch, steps, *value.shape[1:])
            for scale, value in encoded.features.items()
        }
        temporal = self.temporal(features, condition_present=condition_present)
        if isinstance(temporal, tuple):
            temporal = temporal[0]
        return self.decoder(temporal), {"features": temporal}
