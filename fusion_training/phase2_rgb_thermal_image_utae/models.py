"""Asynchronous multimodal U-TAE segmentation model."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from phase1a_rgb_encoder.models import RGBEncoder, RGBSegmentationDecoder
from phase1b_rgb_utae.models import UTAETemporalAggregator
from phase1c_thermal_image_encoder.models import ImageEncoder


class MaskedModalityFusion(nn.Module):
    def __init__(self, channels: int, modality_count: int = 2):
        super().__init__()
        self.modality_count = modality_count
        self.score = nn.ModuleList([nn.Conv2d(channels + 2, 1, 1) for _ in range(modality_count)])
        self.embedding = nn.Parameter(torch.zeros(modality_count, channels))
        nn.init.normal_(self.embedding, std=0.02)

    def forward(self, values, available, counts, ages):
        valid_values = []
        scores = []
        for index, (value, valid, count, age) in enumerate(zip(values, available, counts, ages)):
            valid = valid.to(value.dtype)
            count = torch.log1p(count).to(value.dtype)
            age = torch.log1p(age).to(value.dtype)
            value = value + self.embedding[index][None, :, None, None]
            valid_values.append(value * valid)
            scores.append(self.score[index](torch.cat((value, count, age), dim=1)))
        score = torch.cat(scores, dim=1)
        valid = torch.cat([item.to(torch.bool) for item in available], dim=1)
        weights = _masked_softmax(score, valid, dim=1)
        fused = sum(value * weights[:, index:index + 1] for index, value in enumerate(valid_values))
        return fused, valid.any(dim=1, keepdim=True), weights


def _masked_softmax(logits, valid, dim):
    valid = torch.broadcast_to(valid.bool(), logits.shape)
    values = logits.float().masked_fill(~valid, -torch.inf)
    any_valid = valid.any(dim=dim, keepdim=True)
    values = torch.where(any_valid, values, torch.zeros_like(values))
    weights = torch.softmax(values, dim=dim)
    return torch.where(valid, weights, torch.zeros_like(weights))


class AsyncUTAESegmenter(nn.Module):
    """Raw-image asynchronous RGB/thermal model with independent histories."""

    def __init__(self, rgb_encoder=None, thermal_encoder=None, decoder=None, freeze_encoders=False):
        super().__init__()
        self.rgb_encoder = rgb_encoder or RGBEncoder(pretrained=False)
        self.thermal_encoder = thermal_encoder or ImageEncoder(pretrained=False, input_channels=3)
        self.rgb_temporal = UTAETemporalAggregator()
        self.thermal_temporal = UTAETemporalAggregator()
        self.fusion = nn.ModuleDict({str(scale): MaskedModalityFusion(channels) for scale, channels in {
            2: 32, 4: 64, 8: 128
        }.items()})
        self.decoder = decoder or RGBSegmentationDecoder()
        self.freeze_encoders = freeze_encoders
        self.freeze_temporal = False

    def freeze_pretrained_branches(self):
        """Freeze both pretrained modality encoders and temporal aggregators."""
        for module in (self.rgb_encoder, self.thermal_encoder,
                       self.rgb_temporal, self.thermal_temporal):
            module.requires_grad_(False)
            module.eval()
        self.freeze_encoders = True
        self.freeze_temporal = True

    def _branch(self, sequence, encoder, temporal, frame_valid, feature_valid, time_features, return_details):
        if sequence is None or not frame_valid.any():
            return None
        batch, steps = sequence.shape[:2]
        frames = sequence.reshape(batch * steps, *sequence.shape[2:])
        branch_frozen = self.freeze_encoders and self.freeze_temporal
        context = torch.no_grad() if branch_frozen else torch.enable_grad()
        with context:
            encoded = encoder(frames)
        features = {
            int(scale.split("/")[-1]): value.reshape(batch, steps, *value.shape[1:])
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
        with context:
            result = temporal(features, frame_valid, feature_valid, time_features, return_attention=True)
        return result

    def forward(
        self,
        rgb=None,
        thermal=None,
        rgb_frame_valid=None,
        thermal_frame_valid=None,
        rgb_feature_valid=None,
        thermal_feature_valid=None,
        rgb_time_features=None,
        thermal_time_features=None,
        rgb_ages=None,
        thermal_ages=None,
        return_details=False,
    ):
        if rgb is None and thermal is None:
            raise ValueError("at least one modality must contain an observation")
        source = rgb if rgb is not None else thermal
        batch, _, _, height, width = source.shape
        device = source.device
        rgb_frame_valid = rgb_frame_valid if rgb_frame_valid is not None else torch.ones(batch, rgb.shape[1], dtype=torch.bool, device=device) if rgb is not None else None
        thermal_frame_valid = thermal_frame_valid if thermal_frame_valid is not None else torch.ones(batch, thermal.shape[1], dtype=torch.bool, device=device) if thermal is not None else None
        branches = []
        details = {}
        if rgb is not None and rgb_frame_valid.any():
            branches.append(self._branch(rgb, self.rgb_encoder, self.rgb_temporal, rgb_frame_valid, rgb_feature_valid, rgb_time_features, return_details))
        if thermal is not None and thermal_frame_valid.any():
            branches.append(self._branch(thermal, self.thermal_encoder, self.thermal_temporal, thermal_frame_valid, thermal_feature_valid, thermal_time_features, return_details))
        # Keep modality metadata aligned with the branches that actually exist.
        ages = []
        for value, frame_valid in ((rgb_ages, rgb_frame_valid), (thermal_ages, thermal_frame_valid)):
            if value is None:
                ages.append(torch.zeros(batch, 1, height, width, device=device))
            elif value.ndim == 2:
                ages.append(value[:, :, None, None].expand(batch, 1, height, width))
            else:
                ages.append(value)
        values, availabilities, counts = [], [], []
        modality_ages = []
        branch_index = 0
        for source_value, frame_valid, age in ((rgb, rgb_frame_valid, ages[0]), (thermal, thermal_frame_valid, ages[1])):
            if source_value is not None and frame_valid.any():
                aggregate, availability, branch_details = branches[branch_index]
                branch_index += 1
                values.append(aggregate)
                availabilities.append(availability)
                counts.append(next(iter(branch_details["counts"].values())).sum(dim=1, keepdim=True))
                modality_ages.append(age)
                details["rgb" if source_value is rgb else "thermal"] = branch_details
        fused = {}
        output_valid = {}
        fusion_weights = {}
        for scale in (2, 4, 8):
            scale_values = [value[scale] for value in values]
            scale_available = [value[scale] for value in availabilities]
            scale_counts = [
                F.interpolate(count, size=value.shape[-2:], mode="nearest")
                for count, value in zip(counts, scale_values)
            ]
            scale_ages = [age.to(scale_values[0].dtype) if age.shape[-2:] == scale_values[0].shape[-2:] else F.interpolate(age, size=scale_values[0].shape[-2:], mode="nearest") for age in modality_ages]
            fused[scale], output_valid[scale], fusion_weights[scale] = self.fusion[str(scale)](
                scale_values, scale_available, scale_counts, scale_ages
            )
        logits = self.decoder({f"1/{scale}": value for scale, value in fused.items()})
        details.update({"features": fused, "output_valid": output_valid, "fusion_weights": fusion_weights})
        return (logits, details) if return_details else logits
