"""Three-scale temporal aggregation over the Phase 1A feature contract."""

from __future__ import annotations

import torch
from torch import nn

from .convlstm import ConvLSTM


DEFAULT_SCALES = {
    "1/2": {"input_channels": 32, "hidden_channels": 32, "kernel_size": 3},
    "1/4": {"input_channels": 64, "hidden_channels": 64, "kernel_size": 3},
    "1/8": {"input_channels": 128, "hidden_channels": 128, "kernel_size": 3},
}


class RGBTemporalEncoder(nn.Module):
    """Apply one independent ConvLSTM to each Phase 1A feature scale."""

    def __init__(self, scales=None, use_condition_present: bool = False, return_all_states: bool = False):
        super().__init__()
        configs = scales or DEFAULT_SCALES
        self.return_all_states = return_all_states
        self.temporal = nn.ModuleDict({
            scale: ConvLSTM(
                input_channels=int(config["input_channels"]),
                hidden_channels=int(config["hidden_channels"]),
                kernel_size=int(config.get("kernel_size", 3)),
                bias=bool(config.get("bias", True)),
                use_condition_present=use_condition_present,
                forget_gate_bias=float(config.get("forget_gate_bias", 0.0)),
            )
            for scale, config in configs.items()
        })
        self.use_condition_present = use_condition_present

    def forward(self, feature_sequence, condition_present=None):
        expected = set(self.temporal)
        if set(feature_sequence) != expected:
            raise ValueError(f"feature_sequence must contain exactly {sorted(expected)}")
        outputs = {}
        diagnostics = {}
        for scale, recurrent in self.temporal.items():
            result = recurrent(
                feature_sequence[scale],
                return_all_states=self.return_all_states,
                condition_present=condition_present,
            )
            if self.return_all_states:
                outputs[scale], diagnostics[scale] = result
            else:
                outputs[scale] = result
        return (outputs, diagnostics) if self.return_all_states else outputs


class TemporalMeanEncoder(nn.Module):
    """Non-recurrent baseline with the same three-scale output contract."""

    def __init__(self, use_condition_present: bool = False):
        super().__init__()
        self.use_condition_present = use_condition_present

    def forward(self, feature_sequence, condition_present=None):
        if self.use_condition_present:
            if condition_present is None:
                condition_present = torch.ones(
                    feature_sequence["1/2"].shape[:2],
                    device=feature_sequence["1/2"].device,
                    dtype=torch.bool,
                )
            mask = condition_present.to(dtype=feature_sequence["1/2"].dtype)
            outputs = {}
            for scale, features in feature_sequence.items():
                weights = mask[:, :, None, None, None].to(features.dtype)
                outputs[scale] = (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            return outputs
        return {scale: features.mean(dim=1) for scale, features in feature_sequence.items()}
