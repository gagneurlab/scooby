"""Scooby-2 (internally scAG): Scooby with an AlphaGenome backbone.

Single-cell RNA and ATAC profile prediction from DNA sequence, using
alphagenome-pytorch as the sequence model and Scooby's cell-state decoder.

``ScoobyAlphaGenome`` is also available as ``scooby.ScoobyAlphaGenome``.
"""

from .config import DataConfig, LoraScope, ModelConfig, ScAGConfig, TrainingConfig
from .lora import (
    StandardizedConvLoRA,
    apply_lora,
    build_lora_targets,
    merge_standardized_conv_lora,
)
from .model import ScoobyAlphaGenome, unwrap
from .predict import predict_with_rc

__all__ = [
    "DataConfig",
    "LoraScope",
    "ModelConfig",
    "ScAGConfig",
    "TrainingConfig",
    "ScoobyAlphaGenome",
    "StandardizedConvLoRA",
    "apply_lora",
    "build_lora_targets",
    "merge_standardized_conv_lora",
    "predict_with_rc",
    "unwrap",
]
