"""Spatial RGB encoder and segmentation training graph."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
import torch.nn.functional as F


@dataclass
class RGBEncoderOutput:
    features: dict[str, torch.Tensor]
    valid: dict[str, torch.Tensor]
    timestamp: object = None


class RGBEncoder(nn.Module):
    scales = ("1/2", "1/4", "1/8")

    def __init__(self, pretrained: bool = False, validity_threshold: float = 0.5):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        old = backbone.conv1
        conv1 = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            conv1.weight[:, :3] = old.weight
            conv1.weight[:, 3].zero_()
        self.conv1, self.bn1, self.relu = conv1, backbone.bn1, backbone.relu
        self.maxpool, self.layer1, self.layer2 = backbone.maxpool, backbone.layer1, backbone.layer2
        self.projections = nn.ModuleDict({"1/2": nn.Conv2d(64, 32, 1), "1/4": nn.Conv2d(64, 64, 1), "1/8": nn.Conv2d(128, 128, 1)})
        self.validity_threshold = validity_threshold

    @classmethod
    def from_pretrained(cls, checkpoint, config=None, map_location="cpu"):
        payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
        config = config or payload.get("config", {})
        encoder = cls(pretrained=False, validity_threshold=config.get("validity_threshold", 0.5))
        encoder.load_state_dict(payload.get("model_state_dict", payload))
        return encoder

    def forward(self, rgb: torch.Tensor, valid_mask: torch.Tensor | None = None, timestamp=None) -> RGBEncoderOutput:
        if rgb.shape[1] == 3:
            if valid_mask is None:
                valid_mask = torch.ones(rgb.shape[0], 1, rgb.shape[2], rgb.shape[3], device=rgb.device, dtype=rgb.dtype)
            rgb = torch.cat((rgb, valid_mask.to(rgb.dtype)), dim=1)
        if rgb.shape[1] != 4:
            raise ValueError("RGBEncoder expects 3 RGB channels plus one validity channel")
        stem = self.relu(self.bn1(self.conv1(rgb)))
        quarter = self.layer1(self.maxpool(stem))
        eighth = self.layer2(quarter)
        features = {"1/2": self.projections["1/2"](stem), "1/4": self.projections["1/4"](quarter), "1/8": self.projections["1/8"](eighth)}
        source_valid = rgb[:, 3:4].float()
        valid = {}
        for scale, feature in features.items():
            factor = int(scale.split("/")[1])
            coverage = F.avg_pool2d(source_valid, kernel_size=factor, stride=factor)
            valid[scale] = coverage >= self.validity_threshold
            if valid[scale].shape[-2:] != feature.shape[-2:]:
                valid[scale] = F.interpolate(valid[scale].float(), size=feature.shape[-2:], mode="nearest").bool()
        return RGBEncoderOutput(features, valid, timestamp)


class DecoderBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class RGBSegmentationDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.b8 = DecoderBlock(128)
        self.to4 = nn.Conv2d(128, 64, 1)
        self.b4 = DecoderBlock(64)
        self.to2 = nn.Conv2d(64, 32, 1)
        self.b2 = DecoderBlock(32)
        self.full = DecoderBlock(32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.b8(features["1/8"])
        x = self.to4(F.interpolate(x, size=features["1/4"].shape[-2:], mode="bilinear", align_corners=False)) + features["1/4"]
        x = self.b4(x)
        x = self.to2(F.interpolate(x, size=features["1/2"].shape[-2:], mode="bilinear", align_corners=False)) + features["1/2"]
        x = self.b2(x)
        x = self.full(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        return self.out(x)


class RGBSegmentationModel(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.encoder = RGBEncoder(pretrained=pretrained)
        self.decoder = RGBSegmentationDecoder()

    def forward(self, rgb, valid_mask=None, timestamp=None):
        encoded = self.encoder(rgb, valid_mask, timestamp)
        return self.decoder(encoded.features), encoded
