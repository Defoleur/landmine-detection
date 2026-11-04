import torch
from torch import nn

from phase2_rgb_thermal_image_utae.models import AsyncUTAESegmenter
from phase2_rgb_thermal_image_utae.scripts.train_fusion_utae import load_module_checkpoint


def test_load_module_checkpoint_accepts_standalone_and_prefixed(tmp_path):
    source = nn.Linear(3, 2)
    standalone = tmp_path / "standalone.pt"
    prefixed = tmp_path / "prefixed.pt"
    torch.save({"model_state_dict": source.state_dict()}, standalone)
    torch.save({
        "model_state_dict": {f"temporal.{key}": value for key, value in source.state_dict().items()}
    }, prefixed)

    standalone_target = nn.Linear(3, 2)
    prefixed_target = nn.Linear(3, 2)
    load_module_checkpoint(standalone_target, standalone)
    load_module_checkpoint(prefixed_target, prefixed, "temporal.")

    assert torch.equal(standalone_target.weight, source.weight)
    assert torch.equal(prefixed_target.bias, source.bias)


def test_freeze_pretrained_branches_leaves_fusion_trainable():
    model = AsyncUTAESegmenter()
    model.freeze_pretrained_branches()

    frozen = (model.rgb_encoder, model.thermal_encoder,
              model.rgb_temporal, model.thermal_temporal)
    assert all(not parameter.requires_grad for module in frozen for parameter in module.parameters())
    assert all(not module.training for module in frozen)
    assert any(parameter.requires_grad for parameter in model.fusion.parameters())
    assert any(parameter.requires_grad for parameter in model.decoder.parameters())
