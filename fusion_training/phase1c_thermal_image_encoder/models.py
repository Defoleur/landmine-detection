from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


@dataclass
class ImageEncoderOutput:
    features: dict[str, torch.Tensor]


class ImageEncoder(nn.Module):
    scales = ("1/2", "1/4", "1/8")

    def __init__(self, pretrained=False):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.conv1, self.bn1, self.relu = backbone.conv1, backbone.bn1, backbone.relu
        self.maxpool, self.layer1, self.layer2 = backbone.maxpool, backbone.layer1, backbone.layer2
        self.projections = nn.ModuleDict({"1/2": nn.Conv2d(64, 32, 1), "1/4": nn.Conv2d(64, 64, 1),
                                          "1/8": nn.Conv2d(128, 128, 1)})

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [B,3,H,W]")
        stem = self.relu(self.bn1(self.conv1(image)))
        quarter = self.layer1(self.maxpool(stem))
        eighth = self.layer2(quarter)
        return ImageEncoderOutput({"1/2": self.projections["1/2"](stem),
                                   "1/4": self.projections["1/4"](quarter),
                                   "1/8": self.projections["1/8"](eighth)})


class ImageSegmentationDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.b8 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.to4, self.b4 = nn.Conv2d(128, 64, 1), nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.to2, self.b2 = nn.Conv2d(64, 32, 1), nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.full = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, features):
        x = self.b8(features["1/8"])
        x = self.to4(F.interpolate(x, size=features["1/4"].shape[-2:], mode="bilinear", align_corners=False)) + features["1/4"]
        x = self.b4(x)
        x = self.to2(F.interpolate(x, size=features["1/2"].shape[-2:], mode="bilinear", align_corners=False)) + features["1/2"]
        x = self.b2(x)
        return self.out(self.full(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)))


class ImageSegmentationModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.encoder = ImageEncoder(pretrained)
        self.decoder = ImageSegmentationDecoder()

    def forward(self, image):
        encoded = self.encoder(image)
        return self.decoder(encoded.features), encoded
