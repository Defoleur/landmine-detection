import torch

from phase1d_thermal_image_utae.temporal_models import ThermalImageUTAE


def test_three_channel_thermal_singleton():
    model = ThermalImageUTAE().eval()
    image = torch.randn(1, 1, 3, 64, 64)
    valid = torch.ones(1, 1, dtype=torch.bool)
    with torch.no_grad():
        logits, details = model(image, valid, return_details=True)
    assert logits.shape == (1, 1, 64, 64)
    assert details["availability"][8].all()


def test_thermal_padding_does_not_change_prediction():
    model = ThermalImageUTAE().eval()
    image = torch.randn(1, 2, 3, 64, 64)
    valid = torch.tensor([[True, False]])
    altered = image.clone()
    altered[:, 1] = 500
    with torch.no_grad():
        first = model(image, valid)
        second = model(altered, valid)
    assert torch.allclose(first, second, atol=1e-5)
