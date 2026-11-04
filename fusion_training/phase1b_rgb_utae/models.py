"""U-TAE-derived masked temporal attention for RGB feature histories."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


SCALES = (2, 4, 8)
DEFAULT_INPUT_CHANNELS = {2: 32, 4: 64, 8: 128}


def masked_softmax(logits: torch.Tensor, valid: torch.Tensor, dim: int) -> torch.Tensor:
    """Softmax over valid entries, returning zeros for all-invalid slices."""
    valid = torch.broadcast_to(valid.bool(), logits.shape)
    values = logits.float().masked_fill(~valid, -torch.inf)
    any_valid = valid.any(dim=dim, keepdim=True)
    values = torch.where(any_valid, values, torch.zeros_like(values))
    weights = torch.softmax(values, dim=dim)
    return torch.where(valid, weights, torch.zeros_like(weights))


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels for [B,T,C,H,W] feature histories."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.norm(value.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)


class FeatureAdapter(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.projection = nn.Conv2d(input_channels, output_channels, 1)
        self.norm = ChannelLayerNorm(output_channels)
        self.activation = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, steps, _, height, width = value.shape
        value = self.projection(value.reshape(batch * steps, value.shape[2], height, width))
        value = value.reshape(batch, steps, value.shape[1], height, width)
        return self.activation(self.norm(value))


class UTAETemporalAggregator(nn.Module):
    """Learned-query, spatially independent temporal attention at scale 8."""

    def __init__(
        self,
        input_channels: dict[int, int] | None = None,
        common_channels: dict[int, int] | None = None,
        heads: int = 4,
        key_dim: int = 8,
        time_feature_dim: int = 7,
    ):
        super().__init__()
        input_channels = input_channels or DEFAULT_INPUT_CHANNELS
        common_channels = common_channels or DEFAULT_INPUT_CHANNELS
        if common_channels[8] % heads:
            raise ValueError("scale-8 channels must be divisible by heads")
        self.common_channels = common_channels
        self.heads = heads
        self.key_dim = key_dim
        self.adapters = nn.ModuleDict({
            str(scale): FeatureAdapter(input_channels[scale], common_channels[scale])
            for scale in SCALES
        })
        self.time_mlp = nn.Sequential(
            nn.Linear(time_feature_dim, 64), nn.GELU(), nn.Linear(64, common_channels[8])
        )
        self.time_norm = ChannelLayerNorm(common_channels[8])
        group_channels = common_channels[8] // heads
        self.key_projections = nn.ModuleList(
            [nn.Linear(group_channels, key_dim) for _ in range(heads)]
        )
        self.queries = nn.Parameter(torch.randn(heads, key_dim) * 0.02)

    def _resize_attention(self, attention: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        batch, heads, steps, height, width = attention.shape
        return F.interpolate(
            attention.reshape(batch * heads * steps, 1, height, width),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, heads, steps, *size)

    def _aggregate_scale(
        self,
        features: torch.Tensor,
        attention: torch.Tensor,
        feature_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, steps, channels, height, width = features.shape
        heads = self.heads
        if channels % heads:
            raise ValueError("all common channel widths must be divisible by heads")
        weights = self._resize_attention(attention, (height, width))
        valid = feature_valid.squeeze(2).unsqueeze(1).bool()
        weights = weights * valid.to(weights.dtype)
        denominator = weights.sum(dim=2, keepdim=True)
        uniform = valid.to(weights.dtype) / valid.sum(dim=2, keepdim=True).clamp_min(1)
        weights = torch.where(
            denominator > 0,
            weights / denominator.clamp_min(torch.finfo(weights.dtype).eps),
            uniform.expand_as(weights),
        )
        grouped = torch.nan_to_num(features).reshape(batch, steps, heads, channels // heads, height, width)
        aggregate = (grouped.permute(0, 2, 1, 3, 4, 5) * weights.unsqueeze(3)).sum(dim=2)
        aggregate = aggregate.reshape(batch, channels, height, width)
        available = valid.any(dim=2).any(dim=1, keepdim=True)
        counts = valid[:, :1].sum(dim=2).to(aggregate.dtype)
        aggregate = aggregate * available.to(aggregate.dtype)
        return aggregate, available, counts

    def forward(
        self,
        features: dict[int, torch.Tensor],
        frame_valid: torch.Tensor,
        feature_valid: dict[int, torch.Tensor] | None = None,
        time_features: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        if set(features) != set(SCALES):
            raise ValueError(f"features must contain scales {SCALES}")
        reference = features[8]
        batch, steps = reference.shape[:2]
        frame_valid = frame_valid.bool()
        if frame_valid.shape != (batch, steps):
            raise ValueError("frame_valid must have shape [B,T]")
        feature_valid = feature_valid or {
            scale: frame_valid[:, :, None, None, None].expand(
                batch, steps, 1, features[scale].shape[-2], features[scale].shape[-1]
            )
            for scale in SCALES
        }
        adapted = {scale: self.adapters[str(scale)](features[scale]) for scale in SCALES}
        if time_features is None:
            time_features = reference.new_zeros(batch, steps, self.time_mlp[0].in_features)
        temporal = self.time_mlp(time_features.to(dtype=reference.dtype))
        scale8 = self.time_norm(adapted[8] + temporal[:, :, :, None, None])
        # Keep the explicit validity mask in the score computation; padded features never rank.
        groups = scale8.reshape(batch, steps, self.heads, self.common_channels[8] // self.heads,
                                scale8.shape[-2], scale8.shape[-1])
        scores = []
        for head, projection in enumerate(self.key_projections):
            keys = projection(groups[:, :, head].permute(0, 1, 3, 4, 2))
            scores.append((keys * self.queries[head]).sum(dim=-1) / math.sqrt(self.key_dim))
        score = torch.stack(scores, dim=1)
        valid8 = feature_valid[8].squeeze(2).unsqueeze(1) & frame_valid[:, None, :, None, None]
        attention = masked_softmax(score, valid8, dim=2)
        aggregates, availability, counts = {}, {}, {}
        for scale in SCALES:
            value = scale8 if scale == 8 else adapted[scale]
            aggregates[scale], availability[scale], counts[scale] = self._aggregate_scale(
                value, attention, feature_valid[scale] & frame_valid[:, :, None, None, None]
            )
        details = {"attention": attention, "counts": counts, "adapted": adapted}
        return (aggregates, availability, details) if return_attention else (aggregates, availability)


class MaskedMeanAggregator(UTAETemporalAggregator):
    """Order-independent temporal mean baseline with the same feature adapters."""

    def forward(self, features, frame_valid, feature_valid=None, time_features=None,
                return_attention=False):
        if set(features) != set(SCALES):
            raise ValueError(f"features must contain scales {SCALES}")
        reference = features[8]
        batch, steps = reference.shape[:2]
        frame_valid = frame_valid.bool()
        feature_valid = feature_valid or {
            scale: frame_valid[:, :, None, None, None].expand(
                batch, steps, 1, features[scale].shape[-2], features[scale].shape[-1]
            )
            for scale in SCALES
        }
        adapted = {scale: self.adapters[str(scale)](features[scale]) for scale in SCALES}
        valid8 = feature_valid[8] & frame_valid[:, :, None, None, None]
        attention = valid8.squeeze(2).unsqueeze(1).expand(
            batch, self.heads, steps, reference.shape[-2], reference.shape[-1]
        ).to(reference.dtype)
        attention = attention / attention.sum(dim=2, keepdim=True).clamp_min(1)
        aggregates, availability, counts = {}, {}, {}
        for scale in SCALES:
            aggregates[scale], availability[scale], counts[scale] = self._aggregate_scale(
                adapted[scale], attention, feature_valid[scale] & frame_valid[:, :, None, None, None]
            )
        details = {"attention": attention, "counts": counts, "adapted": adapted}
        return (aggregates, availability, details) if return_attention else (aggregates, availability)


class RGBUTAE(nn.Module):
    """RGB encoder plus U-TAE temporal aggregation and segmentation decoder."""

    def __init__(self, encoder, decoder, temporal=None, freeze_encoder=False):
        super().__init__()
        self.encoder = encoder
        self.temporal = temporal or UTAETemporalAggregator()
        self.decoder = decoder
        self.freeze_encoder = freeze_encoder

    def forward(self, rgb, frame_valid, time_features=None, feature_valid=None, return_details=False):
        batch, steps = rgb.shape[:2]
        frames = rgb.reshape(batch * steps, *rgb.shape[2:])
        context = torch.no_grad() if self.freeze_encoder else torch.enable_grad()
        with context:
            encoded = self.encoder(frames)
        features = {
            int(scale.split("/")[-1]) if isinstance(scale, str) else scale:
            value.reshape(batch, steps, *value.shape[1:])
            for scale, value in encoded.features.items()
        }
        if feature_valid is None:
            encoded_valid = getattr(encoded, "valid", {})
            feature_valid = {}
            for scale, value in features.items():
                valid = encoded_valid.get(f"1/{scale}")
                if valid is None:
                    valid = frame_valid[:, :, None, None, None].expand(
                        batch, steps, 1, value.shape[-2], value.shape[-1]
                    )
                else:
                    valid = valid.reshape(batch, steps, *valid.shape[1:])
                feature_valid[scale] = valid & frame_valid[:, :, None, None, None]
        result = self.temporal(
            features, frame_valid, feature_valid, time_features, return_attention=return_details
        )
        if return_details:
            aggregate, availability, temporal_details = result
        else:
            aggregate, availability = result
            temporal_details = {}
        logits = self.decoder({f"1/{scale}": value for scale, value in aggregate.items()})
        if return_details:
            return logits, {"features": aggregate, "availability": availability, **temporal_details}
        return logits
