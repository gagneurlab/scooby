"""Tests for the model and its checkpoint round-trip.

Building an AlphaGenome backbone allocates the full parameter set, so these
are marked slow and skipped by default. Run them with ``pytest -m slow``.
"""

import pytest
import torch

from scooby.scag.config import LoraScope, ModelConfig, ScAGConfig, TrainingConfig
from scooby.scag.lora import StandardizedConvLoRA
from scooby.scag.model import ScoobyAlphaGenome, unwrap

pytestmark = pytest.mark.slow

CELL_EMB_DIM = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 16_384


def model_config(**overrides) -> ModelConfig:
    kwargs = dict(
        cell_emb_dim=CELL_EMB_DIM,
        num_tracks=3,
        target_bins=None,
        early_exit_32bp=True,
        use_lora=True,
        lora=LoraScope(encoder_convs=True, decoder_convs=False, attn=True, mlp=True),
        lora_r=2,
        lora_alpha=2,
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


def run_config(model_cfg, tmp_path) -> ScAGConfig:
    return ScAGConfig(
        output_dir=str(tmp_path),
        run_name="test",
        model=model_cfg,
        training=TrainingConfig(),
    )


@pytest.fixture(scope="module")
def inputs():
    torch.manual_seed(0)
    sequence = torch.nn.functional.one_hot(
        torch.randint(0, 4, (1, SEQ_LEN)), num_classes=4
    ).float().to(DEVICE)
    cell_emb = torch.randn(1, 4, CELL_EMB_DIM, device=DEVICE)
    return sequence, cell_emb


def test_forward_shape(inputs):
    sequence, cell_emb = inputs
    model = ScoobyAlphaGenome(
        cell_emb_dim=CELL_EMB_DIM, n_tracks=3, target_bins=None, early_exit_32bp=True
    ).to(DEVICE).eval()
    with torch.no_grad():
        out = model(sequence, cell_emb)
    assert out.shape == (1, SEQ_LEN // 32, 4 * 3)
    assert torch.isfinite(out).all()


def test_round_trip_preserves_predictions(tmp_path, inputs):
    """A saved model reloads to the same weights and the same outputs."""
    from scooby.scag.lora import apply_lora

    sequence, cell_emb = inputs
    cfg = run_config(model_config(), tmp_path)

    model = ScoobyAlphaGenome(disable_cache=True, **cfg.model.model_kwargs())
    wrapped = apply_lora(model, cfg.model, train=True).to(DEVICE)
    with torch.no_grad():  # move the adapters off their zero init
        for module in wrapped.modules():
            if isinstance(module, StandardizedConvLoRA):
                module.lora_B.normal_(std=0.02)
    wrapped.eval()
    with torch.no_grad():
        expected = wrapped(sequence, cell_emb)

    checkpoint = tmp_path / "ckpt"
    unwrap(wrapped).save_pretrained(str(checkpoint), config=cfg)

    reloaded = ScoobyAlphaGenome.from_pretrained(str(checkpoint), device=DEVICE)
    with torch.no_grad():
        actual = reloaded(sequence, cell_emb)
    torch.testing.assert_close(actual, expected)


def test_from_pretrained_rejects_a_mismatched_config(tmp_path, inputs):
    cfg = run_config(model_config(), tmp_path)
    model = ScoobyAlphaGenome(disable_cache=True, **cfg.model.model_kwargs())
    checkpoint = tmp_path / "ckpt"
    model.save_pretrained(str(checkpoint), config=cfg)

    with pytest.raises(RuntimeError, match="does not match"):
        ScoobyAlphaGenome.from_pretrained(str(checkpoint), cell_emb_dim=CELL_EMB_DIM * 2)


def test_head_input_norm_bounds_the_decoder_input(inputs):
    """Normalising the embedding keeps the decoder's input scale fixed."""
    sequence, _ = inputs
    model = ScoobyAlphaGenome(
        cell_emb_dim=CELL_EMB_DIM, target_bins=None,
        early_exit_32bp=True, head_input_norm=True,
    ).to(DEVICE).eval()
    with torch.no_grad():
        emb = model.forward_seq_to_emb(sequence)
    assert emb.std().item() == pytest.approx(1.0, abs=0.1)
