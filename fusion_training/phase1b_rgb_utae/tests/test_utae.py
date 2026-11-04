import torch

from phase1b_rgb_utae.models import UTAETemporalAggregator, masked_softmax


def _features(batch=4, steps=7):
    return {
        2: torch.randn(batch, steps, 32, 8, 8),
        4: torch.randn(batch, steps, 64, 4, 4),
        8: torch.randn(batch, steps, 128, 2, 2),
    }


def test_unequal_histories_and_masked_attention():
    model = UTAETemporalAggregator().eval()
    features = _features()
    frame_valid = torch.tensor([
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
    ], dtype=torch.bool)
    with torch.no_grad():
        aggregate, available, details = model(features, frame_valid, return_attention=True)
    assert details["attention"].shape == (4, 4, 7, 2, 2)
    invalid = details["attention"].masked_select(~frame_valid[:, None, :, None, None])
    assert torch.equal(invalid, torch.zeros_like(invalid))
    assert all(value.shape[0] == 4 for value in aggregate.values())
    assert all(value.dtype == torch.bool for value in available.values())


def test_padding_invariance():
    model = UTAETemporalAggregator().eval()
    features = _features(batch=1, steps=3)
    valid = torch.tensor([[True, True, False]])
    altered = {scale: value.clone() for scale, value in features.items()}
    for value in altered.values():
        value[:, 2] = 1000
    with torch.no_grad():
        first = model(features, valid)[0]
        second = model(altered, valid)[0]
    for scale in first:
        assert torch.allclose(first[scale], second[scale], atol=1e-6)


def test_all_invalid_softmax_is_finite_and_zero():
    logits = torch.randn(2, 3, requires_grad=True)
    weights = masked_softmax(logits, torch.zeros_like(logits, dtype=torch.bool), dim=1)
    assert torch.equal(weights, torch.zeros_like(weights))
    weights.sum().backward()
    assert torch.isfinite(logits.grad).all()
