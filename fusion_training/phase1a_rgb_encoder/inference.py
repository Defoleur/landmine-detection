"""Decoder-free feature extraction API."""

import torch

from .models import RGBEncoder


def load_encoder(checkpoint, config=None, device="cpu"):
    encoder = RGBEncoder(pretrained=False)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("model_state_dict", payload)
    encoder.load_state_dict(state)
    return encoder.to(device).eval()


@torch.inference_mode()
def extract_features(encoder, rgb, valid_mask=None, timestamp=None):
    output = encoder(rgb, valid_mask, timestamp)
    return {"features": output.features, "features_valid": output.valid, "timestamp": output.timestamp}
