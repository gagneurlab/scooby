"""LoRA adaptation for the AlphaGenome backbone.

Transformer ``Linear`` layers are adapted with peft. Convolutions use
:class:`StandardizedConvLoRA`, because a parallel adapter branch on
AlphaGenome's weight-standardized convolutions is not scale-controlled.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # AlphaGenome is a hard dependency at runtime, optional at import time.
    from alphagenome_pytorch.convolutions import StandardizedConv1d
except ImportError:  # pragma: no cover
    StandardizedConv1d = ()


# ----------------------------------------------------------------------
# Target selection
# ----------------------------------------------------------------------
LORA_REGION_REGEXES: Dict[str, str] = {
    "encoder_convs": r"alphagenome\.encoder\.down_blocks\.\d+\.block[12]\.conv",
    "decoder_convs": r"alphagenome\.decoder\.up_blocks\.\d+\.(conv_in|pointwise|conv_out)\.conv",
    "attn": r"alphagenome\.tower\.blocks\.\d+\.mha\.(q_proj|k_proj|v_proj)",
    "mlp": r"alphagenome\.tower\.blocks\.\d+\.mlp\.(fc1|fc2)",
}

CONV_REGIONS = ("encoder_convs", "decoder_convs")

# The pair-representation branch is never adapted.


def build_lora_targets(
    encoder_convs: bool = True,
    decoder_convs: bool = False,
    attn: bool = True,
    mlp: bool = True,
    regions: Optional[Iterable[str]] = None,
) -> str:
    """Compose a ``target_modules`` regex from per-region flags.

    Args:
        encoder_convs: Adapt ``encoder.down_blocks[*].block{1,2}.conv``.
        decoder_convs: Adapt the convolutions in ``decoder.up_blocks[*]``.
            With ``early_exit_32bp`` the decoder is trimmed to two blocks,
            so most of these adapters would be unused.
        attn: Adapt ``mha.q_proj/k_proj/v_proj`` in every transformer block.
        mlp: Adapt ``mlp.fc1/fc2`` in every transformer block.
        regions: Restrict the output to this subset of region names, used to
            split targets between the two mechanisms.

    Returns:
        A regex for ``LoraConfig(target_modules=...)``.

    Raises:
        ValueError: if no region is selected.
    """
    flags = {
        "encoder_convs": encoder_convs,
        "decoder_convs": decoder_convs,
        "attn": attn,
        "mlp": mlp,
    }
    if regions is not None:
        flags = {k: v for k, v in flags.items() if k in set(regions)}
    parts = [LORA_REGION_REGEXES[k] for k, on in flags.items() if on]
    if not parts:
        raise ValueError("no LoRA regions selected")
    return "|".join(parts)


# ----------------------------------------------------------------------
# Convolutional adapters
# ----------------------------------------------------------------------
class StandardizedConvLoRA(nn.Module):
    """LoRA for a weight-standardized convolution.

    The low-rank update is applied to the kernel and standardized together
    with it,

    .. code-block:: text

        y = conv(x, standardize(W + (alpha / r) * B A))

    rather than added as a parallel branch, so the layer's gain stays fixed
    however large the adapter grows, and merging ``B A`` into ``W`` is exact.

    Args:
        base_layer: The convolution to adapt. Its parameters are frozen.
        r: Adapter rank.
        alpha: Adapter scaling numerator; the update is scaled by ``alpha/r``.

    Attributes:
        lora_A: ``(r, in_channels, kernel_size)``, initialised as peft does.
        lora_B: ``(out_channels, r)``, initialised to zero so the adapted
            layer starts out identical to the pretrained one.
    """

    def __init__(self, base_layer: nn.Conv1d, r: int = 8, alpha: int = 8):
        super().__init__()
        if r <= 0:
            raise ValueError(f"r must be positive, got {r}")
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.merged = False

        out_channels, in_channels, kernel_size = base_layer.weight.shape
        self.lora_A = nn.Parameter(torch.empty(r, in_channels, kernel_size))
        self.lora_B = nn.Parameter(torch.zeros(out_channels, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        for p in self.base_layer.parameters():
            p.requires_grad = False

    def delta_weight(self) -> torch.Tensor:
        """The low-rank kernel update, shaped like the base kernel."""
        return torch.einsum("or,rik->oik", self.lora_B, self.lora_A) * self.scaling

    def effective_weight(self) -> torch.Tensor:
        w = self.base_layer.weight
        if self.merged:
            return w
        return w + self.delta_weight().to(w.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _standardized_conv1d(self.base_layer, x, self.effective_weight())

    @torch.no_grad()
    def merge(self) -> nn.Conv1d:
        """Fold the adapter into the base kernel and return the base layer."""
        if not self.merged:
            self.base_layer.weight += self.delta_weight().to(self.base_layer.weight.dtype)
            self.merged = True
        return self.base_layer

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, merged={self.merged}"


def _standardized_conv1d(
    module: nn.Conv1d, x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Run ``StandardizedConv1d``'s forward with an externally supplied kernel.

    Mirrors ``alphagenome_pytorch.convolutions.StandardizedConv1d.forward``.
    ``tests/test_lora.py`` asserts the two agree for a zero update.
    """
    mean = weight.mean(dim=(1, 2), keepdim=True)
    var = weight.var(dim=(1, 2), keepdim=True, unbiased=False)
    fan_in = module.in_channels * module.kernel_size[0]
    floor = torch.tensor(1e-4, device=weight.device, dtype=weight.dtype)
    scale_factor = torch.rsqrt(torch.maximum(var * fan_in, floor)) * module.scale
    w = (weight - mean) * scale_factor

    if getattr(module, "pad_mode", None) == "same":
        pad_total = module.kernel_size[0] - 1
        pad_left = pad_total // 2
        x = F.pad(x, (pad_left, pad_total - pad_left))
    return F.conv1d(x, w, module.bias, module.stride, 0, module.dilation, module.groups)


def inject_standardized_conv_lora(
    model: nn.Module, target_regex: str, r: int = 8, alpha: int = 8
) -> List[str]:
    """Replace matching weight-standardized convolutions with adapted ones.

    Modules whose kernel size is 1 are plain ``nn.Conv1d`` and are left
    alone; peft handles those correctly.

    Args:
        model: The model to modify in place.
        target_regex: Regex matched against qualified module names.
        r: Adapter rank.
        alpha: Adapter scaling numerator.

    Returns:
        The names of the modules that were replaced.
    """
    pattern = re.compile(target_regex)
    replaced: List[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, StandardizedConv1d):
            continue
        if not pattern.search(name):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, StandardizedConvLoRA(module, r=r, alpha=alpha))
        replaced.append(name)
    return replaced


def merge_standardized_conv_lora(model: nn.Module) -> int:
    """Fold every :class:`StandardizedConvLoRA` back into its base layer."""
    count = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, StandardizedConvLoRA):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, module.merge())
        count += 1
    return count


# ----------------------------------------------------------------------
# peft path (transformer layers, and convolutions in legacy mode)
# ----------------------------------------------------------------------
def patch_peft_conv_padding(model: nn.Module) -> None:
    """Give peft's convolutional adapters AlphaGenome's SAME padding.

    ``StandardizedConv1d`` leaves the underlying ``nn.Conv1d`` at
    ``padding=0`` and pads inside its forward. peft copies the visible
    padding when it builds ``lora_A``, so for ``kernel_size > 1`` the adapter
    output is shorter than the base output and the addition fails.

    Only needed in legacy mode, where convolutions are adapted by peft.
    """
    if not StandardizedConv1d:  # pragma: no cover
        return
    for module in model.modules():
        base = getattr(module, "base_layer", None)
        if not isinstance(base, StandardizedConv1d):
            continue
        k = base.kernel_size[0]
        if k == 1:
            continue
        if k % 2 == 0:
            raise ValueError(
                f"even kernel size ({k}) needs asymmetric SAME padding, "
                "which peft's adapter cannot express"
            )
        for adapter in getattr(module, "lora_A", {}).values():
            adapter.padding = ((k - 1) // 2,)


def apply_lora(model: nn.Module, model_cfg, train: bool = False):
    """Attach LoRA adapters to a :class:`~scooby.scag.model.ScoobyAlphaGenome`.

    Convolutions are adapted by :func:`inject_standardized_conv_lora` and
    transformer layers by peft, unless ``model_cfg.standardized_conv_lora``
    is False, in which case peft handles both.

    Args:
        model: The unwrapped model.
        model_cfg: A :class:`~scooby.scag.config.ModelConfig`.
        train: Whether to unfreeze the trainable parameters. When False the
            whole model is frozen for inference.

    Returns:
        The peft-wrapped model. The unwrapped model remains reachable as
        ``wrapped.base_model.model``.
    """
    from peft import LoraConfig, get_peft_model

    scope = model_cfg.lora
    explicit = model_cfg.lora_targets is not None
    use_standardized = model_cfg.standardized_conv_lora and not explicit

    if use_standardized:
        conv_on = scope.encoder_convs or scope.decoder_convs
        if conv_on:
            conv_regex = build_lora_targets(
                encoder_convs=scope.encoder_convs,
                decoder_convs=scope.decoder_convs,
                attn=False,
                mlp=False,
                regions=CONV_REGIONS,
            )
            inject_standardized_conv_lora(
                model, conv_regex, r=model_cfg.lora_r, alpha=model_cfg.lora_alpha
            )
        peft_regions = ("attn", "mlp")
        if not (scope.attn or scope.mlp):
            # peft needs at least one target; nothing left for it to do.
            raise ValueError(
                "standardized_conv_lora with only convolutional regions "
                "enabled is not supported; enable attn or mlp, or set "
                "model.standardized_conv_lora: false"
            )
        peft_targets = build_lora_targets(
            encoder_convs=False, decoder_convs=False,
            attn=scope.attn, mlp=scope.mlp, regions=peft_regions,
        )
    else:
        peft_targets = model_cfg.lora_targets or build_lora_targets(
            encoder_convs=scope.encoder_convs,
            decoder_convs=scope.decoder_convs,
            attn=scope.attn,
            mlp=scope.mlp,
        )

    lora_config = LoraConfig(
        target_modules=peft_targets,
        r=model_cfg.lora_r,
        lora_alpha=model_cfg.lora_alpha,
        lora_dropout=model_cfg.lora_dropout,
    )
    wrapped = get_peft_model(model, lora_config)
    if not use_standardized:
        patch_peft_conv_padding(wrapped)

    inner = wrapped.base_model.model
    if train:
        trainable = [inner.cell_state_to_conv]
        if getattr(inner, "use_transform_emb", False):
            trainable.append(inner.transform_emb)
        # The early-exit path adds a freshly initialised 32 bp embedder,
        # which has no pretrained weights to preserve.
        if getattr(inner, "early_exit_32bp", False):
            trainable.append(inner.embedder_32bp)
        if getattr(inner, "head_input_norm", None) is not None:
            trainable.append(inner.head_input_norm)
        for module in trainable:
            for p in module.parameters():
                p.requires_grad = True
        for module in wrapped.modules():
            if isinstance(module, StandardizedConvLoRA):
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True
    else:
        for p in wrapped.parameters():
            p.requires_grad = False
    return wrapped


def trainable_parameter_summary(model: nn.Module) -> Tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
