"""Spatial ConvLSTM components for fixed condition-slot sequences."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


State = tuple[torch.Tensor, torch.Tensor]


class ConvLSTMCell(nn.Module):
    """A single spatial ConvLSTM recurrence."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
        forget_gate_bias: float = 0.0,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve spatial size")
        self.hidden_channels = hidden_channels
        self.forget_gate_bias = forget_gate_bias
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )

    def init_state(self, x: torch.Tensor) -> State:
        shape = (x.shape[0], self.hidden_channels, x.shape[-2], x.shape[-1])
        zeros = x.new_zeros(shape)
        return zeros, zeros.clone()

    def forward(self, x: torch.Tensor, state: Optional[State] = None) -> State:
        if x.ndim != 4:
            raise ValueError("ConvLSTMCell input must have shape [B,C,H,W]")
        if state is None:
            h_prev, c_prev = self.init_state(x)
        else:
            h_prev, c_prev = state
            expected = (x.shape[0], self.hidden_channels, x.shape[-2], x.shape[-1])
            if tuple(h_prev.shape) != expected or tuple(c_prev.shape) != expected:
                raise ValueError(f"state tensors must have shape {expected}")

        gates = self.gates(torch.cat((x, h_prev), dim=1))
        input_gate, forget_gate, output_gate, candidate = gates.chunk(4, dim=1)
        if self.forget_gate_bias:
            forget_gate = forget_gate + self.forget_gate_bias
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate = torch.tanh(candidate)
        c_next = forget_gate * c_prev + input_gate * candidate
        h_next = output_gate * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    """Run a ConvLSTM over [B,T,C,H,W] condition features.

    When ``use_condition_present`` is enabled, absent condition slots preserve
    the previous hidden and cell states instead of performing a zero-feature
    update. State is always initialized afresh for each forward call unless an
    explicit initial state is provided.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
        use_condition_present: bool = False,
        forget_gate_bias: float = 0.0,
    ):
        super().__init__()
        self.use_condition_present = use_condition_present
        self.cell = ConvLSTMCell(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            bias=bias,
            forget_gate_bias=forget_gate_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[State] = None,
        return_all_states: bool = False,
        condition_present: Optional[torch.Tensor] = None,
    ):
        if x.ndim != 5:
            raise ValueError("ConvLSTM input must have shape [B,T,C,H,W]")
        batch, steps = x.shape[:2]
        if self.use_condition_present:
            if condition_present is None:
                condition_present = torch.ones(batch, steps, device=x.device, dtype=torch.bool)
            else:
                condition_present = self._validate_condition_present(condition_present, batch, steps, x.device)
        elif condition_present is not None:
            # Keep the non-mask mode equivalent to the standard ConvLSTM.
            condition_present = None

        current = state
        all_states = []
        for step in range(steps):
            previous = current
            candidate = self.cell(x[:, step], current)
            if condition_present is not None:
                present = condition_present[:, step].view(batch, 1, 1, 1)
                current = tuple(
                    torch.where(present, new, old)
                    for new, old in zip(candidate, previous or self.cell.init_state(x[:, step]))
                )
            else:
                current = candidate
            if return_all_states:
                all_states.append(current[0])

        if current is None:
            current = self.cell.init_state(x[:, 0])
        output = current[0]
        if return_all_states:
            return output, torch.stack(all_states, dim=1)
        return output

    @staticmethod
    def _validate_condition_present(mask, batch, steps, device):
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, device=device)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if tuple(mask.shape) != (batch, steps):
            raise ValueError(f"condition_present must have shape {(batch, steps)}")
        return mask.to(device=device, dtype=torch.bool)
