import torch

from phase1b_rgb_convlstm.rgb_temporal_encoder import RGBTemporalEncoder, TemporalMeanEncoder


def _features(batch=2, steps=12):
    return {
        "1/2": torch.randn(batch, steps, 32, 8, 8),
        "1/4": torch.randn(batch, steps, 64, 4, 4),
        "1/8": torch.randn(batch, steps, 128, 2, 2),
    }


def test_three_scale_encoder_contract_with_twelve_conditions():
    model = RGBTemporalEncoder(use_condition_present=True).eval()
    with torch.no_grad():
        outputs = model(_features(), torch.ones(2, 12, dtype=torch.bool))
    assert {scale: tuple(value.shape) for scale, value in outputs.items()} == {
        "1/2": (2, 32, 8, 8), "1/4": (2, 64, 4, 4), "1/8": (2, 128, 2, 2)
    }


def test_condition_aware_mean_ignores_absent_slots():
    features = _features(batch=1, steps=3)
    present = torch.tensor([[True, False, True]])
    outputs = TemporalMeanEncoder(use_condition_present=True)(features, present)
    expected = (features["1/2"][:, 0] + features["1/2"][:, 2]) / 2
    assert torch.allclose(outputs["1/2"], expected)
