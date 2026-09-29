import torch

from phase1c_thermal_image_encoder.data import IMAGE_MEAN, IMAGE_STD
from phase1c_thermal_image_encoder.models import ImageSegmentationModel, ImageEncoder


def test_image_encoder_feature_contract():
    output = ImageEncoder()(torch.randn(2, 3, 128, 128))
    assert output.features["1/2"].shape == (2, 32, 64, 64)
    assert output.features["1/4"].shape == (2, 64, 32, 32)
    assert output.features["1/8"].shape == (2, 128, 16, 16)


def test_image_segmentation_output():
    logits, _ = ImageSegmentationModel()(torch.randn(2, 3, 128, 128))
    assert logits.shape == (2, 1, 128, 128)


def test_image_normalization_contract():
    assert len(IMAGE_MEAN) == 3
    assert len(IMAGE_STD) == 3
