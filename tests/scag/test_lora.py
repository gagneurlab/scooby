"""Tests for the convolutional LoRA adapters."""

import math

import pytest
import torch
import torch.nn as nn

from alphagenome_pytorch.convolutions import StandardizedConv1d

from scooby.scag.lora import (
    StandardizedConvLoRA,
    build_lora_targets,
    inject_standardized_conv_lora,
    merge_standardized_conv_lora,
)


def make_conv(in_c=16, out_c=16, k=5, seed=0):
    torch.manual_seed(seed)
    conv = StandardizedConv1d(in_c, out_c, k, padding="same")
    nn.init.normal_(conv.weight, std=0.05)
    return conv


def test_zero_update_matches_base_layer():
    """With lora_B at its zero init, the adapted layer is the pretrained one."""
    conv = make_conv()
    x = torch.randn(2, 16, 64)
    adapted = StandardizedConvLoRA(conv, r=4)
    torch.testing.assert_close(adapted(x), conv(x))


def test_preserves_sequence_length():
    conv = make_conv(k=5)
    x = torch.randn(2, 16, 64)
    assert StandardizedConvLoRA(conv, r=4)(x).shape == (2, 16, 64)


def test_merge_is_exact():
    """Folding the update into the kernel reproduces the adapted layer.

    This does not hold for a parallel adapter branch on a standardized
    convolution, because standardisation is not linear in the kernel.
    """
    conv = make_conv()
    adapted = StandardizedConvLoRA(conv, r=4)
    with torch.no_grad():
        adapted.lora_B.normal_(std=0.3)
    x = torch.randn(2, 16, 64)

    before = adapted(x)
    merged = adapted.merge()
    torch.testing.assert_close(merged(x), before, rtol=1e-5, atol=1e-6)


def test_gain_is_bounded_as_the_adapter_grows():
    """Output scale stays put however large the update gets.

    This is the property that keeps encoder activations at their pretrained
    scale during fine-tuning. A parallel branch has no such bound, and
    AlphaGenome cannot absorb the drift downstream because RMSBatchNorm
    rescales by a fixed running variance.
    """
    conv = make_conv()
    x = torch.randn(2, 16, 64)
    baseline = conv(x).std().item()

    adapted = StandardizedConvLoRA(conv, r=4)
    scales = []
    for magnitude in (0.1, 1.0, 10.0, 100.0):
        with torch.no_grad():
            torch.manual_seed(0)
            adapted.lora_B.normal_(std=magnitude)
        scales.append(adapted(x).std().item())

    assert max(scales) < 3 * baseline, (
        f"output scale grew with adapter magnitude: {scales} vs base {baseline}"
    )


def test_parallel_branch_grows_without_bound():
    """The behaviour the standardized adapter avoids, stated explicitly."""
    conv = make_conv()
    x = torch.randn(2, 16, 64)
    baseline = conv(x).std().item()

    torch.manual_seed(0)
    lora_A = nn.Conv1d(16, 4, 5, padding=2, bias=False)
    lora_B = nn.Conv1d(4, 16, 1, bias=False)
    with torch.no_grad():
        nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
        lora_B.weight.normal_(std=10.0)
    parallel = conv(x) + lora_B(lora_A(x))

    assert parallel.std().item() > 10 * baseline


def test_injection_replaces_only_matching_convolutions():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = make_conv()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.target = Block()
            self.other = Block()
            self.pointwise = nn.Conv1d(16, 16, 1)

    model = Model()
    replaced = inject_standardized_conv_lora(model, r"target\.conv", r=4)

    assert replaced == ["target.conv"]
    assert isinstance(model.target.conv, StandardizedConvLoRA)
    assert isinstance(model.other.conv, StandardizedConv1d)
    assert isinstance(model.pointwise, nn.Conv1d)

    assert merge_standardized_conv_lora(model) == 1
    assert isinstance(model.target.conv, StandardizedConv1d)


def test_base_layer_is_frozen_and_adapter_is_not():
    adapted = StandardizedConvLoRA(make_conv(), r=4)
    assert not any(p.requires_grad for p in adapted.base_layer.parameters())
    assert adapted.lora_A.requires_grad and adapted.lora_B.requires_grad


def test_gradients_reach_the_adapter():
    adapted = StandardizedConvLoRA(make_conv(), r=4)
    adapted(torch.randn(2, 16, 32)).sum().backward()
    assert adapted.lora_A.grad is not None
    assert adapted.lora_B.grad is not None


def test_build_lora_targets_regions():
    targets = build_lora_targets(encoder_convs=True, decoder_convs=False,
                                 attn=True, mlp=False)
    assert "down_blocks" in targets and "mha" in targets
    assert "up_blocks" not in targets and "mlp" not in targets

    with pytest.raises(ValueError):
        build_lora_targets(False, False, False, False)


def test_compile_prefix_is_stripped_from_checkpoints():
    """Weights saved from a compiled model load into an uncompiled one."""
    from scooby.scag.model import _strip_compile_prefix

    state = {
        "_orig_mod.alphagenome.encoder.weight": torch.zeros(1),
        "_orig_mod.cell_state_to_conv.0.bias": torch.zeros(1),
    }
    stripped = _strip_compile_prefix(state)
    assert set(stripped) == {
        "alphagenome.encoder.weight", "cell_state_to_conv.0.bias"
    }
    # A state dict without the prefix is returned untouched.
    plain = {"alphagenome.encoder.weight": torch.zeros(1)}
    assert _strip_compile_prefix(plain) is plain
