import torch

from phase1b_rgb_convlstm.convlstm import ConvLSTM


def test_fixed_twelve_condition_slots():
    model = ConvLSTM(32, 32).eval()
    x = torch.randn(2, 12, 32, 8, 10)
    with torch.no_grad():
        output, states = model(x, return_all_states=True)
    assert output.shape == (2, 32, 8, 10)
    assert states.shape == (2, 12, 32, 8, 10)
    assert torch.isfinite(output).all()


def test_absent_condition_preserves_state():
    model = ConvLSTM(4, 4, use_condition_present=True).eval()
    x = torch.randn(1, 3, 4, 5, 5)
    present = torch.tensor([[True, False, True]])
    with torch.no_grad():
        output, states = model(x, condition_present=present, return_all_states=True)
        first_only = model(x[:, :1])
    assert torch.equal(states[:, 1], states[:, 0])
    assert torch.allclose(first_only, states[:, 0])
    assert torch.isfinite(output).all()


def test_mask_mode_defaults_to_all_present_and_backpropagates():
    model = ConvLSTM(3, 5, use_condition_present=True)
    x = torch.randn(2, 6, 3, 4, 4, requires_grad=True)
    output = model(x)
    output.square().mean().backward()
    assert output.shape == (2, 5, 4, 4)
    assert model.cell.gates.weight.grad is not None
    assert torch.isfinite(model.cell.gates.weight.grad).all()


def test_disabled_mask_matches_standard_path():
    torch.manual_seed(7)
    standard = ConvLSTM(3, 3)
    torch.manual_seed(7)
    masked = ConvLSTM(3, 3, use_condition_present=True)
    masked.load_state_dict(standard.state_dict())
    x = torch.randn(1, 4, 3, 6, 6)
    present = torch.ones(1, 4, dtype=torch.bool)
    assert torch.allclose(standard(x), masked(x, condition_present=present))
