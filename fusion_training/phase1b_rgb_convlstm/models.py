"""Phase 1B temporal encoders and Phase 1A decoder integration."""

from __future__ import annotations

import torch
from torch import nn

from phase1a_rgb_encoder.models import RGBEncoder, RGBSegmentationDecoder
from .rgb_temporal_encoder import RGBTemporalEncoder, TemporalMeanEncoder


class RGBTemporalSegmentationModel(nn.Module):
    def __init__(self, encoder=None, decoder=None, pretrained=False,
                 use_condition_present=False, temporal_mean=False):
        super().__init__()
        self.encoder = encoder or RGBEncoder(pretrained=pretrained)
        self.temporal = (TemporalMeanEncoder(use_condition_present=use_condition_present)
                         if temporal_mean else RGBTemporalEncoder(use_condition_present=use_condition_present))
        self.decoder = decoder or RGBSegmentationDecoder()
        self.temporal_mean = temporal_mean

    def forward(self, rgb_sequence, rgb_validity_sequence=None, condition_present=None,
                freeze_spatial=False):
        if rgb_sequence.ndim != 5:
            raise ValueError("rgb_sequence must have shape [B,T,C,H,W]")
        batch, steps, channels, height, width = rgb_sequence.shape
        if channels == 3:
            if rgb_validity_sequence is None:
                rgb_validity_sequence = torch.ones(batch, steps, 1, height, width,
                                                   device=rgb_sequence.device, dtype=rgb_sequence.dtype)
            rgb_sequence = torch.cat((rgb_sequence, rgb_validity_sequence.to(rgb_sequence.dtype)), dim=2)
        if rgb_sequence.shape[2] != 4:
            raise ValueError("RGB temporal input expects RGB plus validity")
        frames = rgb_sequence.reshape(batch * steps, 4, height, width)
        context = torch.no_grad() if freeze_spatial else torch.enable_grad()
        with context:
            encoded = self.encoder(frames, frames[:, 3:4])
        feature_sequence = {scale: value.reshape(batch, steps, *value.shape[1:])
                            for scale, value in encoded.features.items()}
        temporal = self.temporal(feature_sequence, condition_present=condition_present)
        if self.temporal_mean:
            features = temporal
        else:
            features = temporal[0] if isinstance(temporal, tuple) else temporal
        logits = self.decoder(features)
        return logits, {"features": features, "feature_validity": encoded.valid}


class CachedRGBTemporalSegmentationModel(nn.Module):
    """Temporal model consuming cached Phase 1A feature sequences."""

    def __init__(self, decoder=None, use_condition_present=False, temporal_mean=False):
        super().__init__()
        self.temporal = (TemporalMeanEncoder(use_condition_present=use_condition_present)
                         if temporal_mean else RGBTemporalEncoder(use_condition_present=use_condition_present))
        self.decoder = decoder or RGBSegmentationDecoder()

    def forward(self, features, condition_present=None):
        temporal = self.temporal(features, condition_present=condition_present)
        if isinstance(temporal, tuple):
            temporal = temporal[0]
        return self.decoder(temporal), {"features": temporal}
