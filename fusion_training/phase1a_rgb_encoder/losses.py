"""Numerically safe masked segmentation losses."""

import torch
import torch.nn.functional as F


def masked_bce_with_logits(logits, target, target_valid, rgb_valid):
    mask = (target_valid.bool() & rgb_valid.bool()).to(logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype), reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)
