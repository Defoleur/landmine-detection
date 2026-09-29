"""Phase 1A reusable RGB spatial encoder."""

from .models import RGBEncoder, RGBEncoderOutput, RGBSegmentationModel
from .losses import masked_bce_with_logits

__all__ = ["RGBEncoder", "RGBEncoderOutput", "RGBSegmentationModel", "masked_bce_with_logits"]
