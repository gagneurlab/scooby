"""Typed configuration for scAG.

Every filesystem path used by training or evaluation is declared here and
resolved once, at load time. Relative paths are interpreted relative to the
directory holding the YAML file, so a config can be moved together with its
data without editing. ``~`` and ``$VARS`` are expanded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ----------------------------------------------------------------------
# Path handling
# ----------------------------------------------------------------------
def resolve_path(value: Optional[str], root: Path) -> Optional[str]:
    """Expand ``~``/env vars and make ``value`` absolute relative to ``root``."""
    if value is None:
        return None
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    p = Path(expanded)
    if not p.is_absolute():
        p = root / p
    return str(p)


def _require_exists(path: Optional[str], label: str, errors: List[str]) -> None:
    if path is not None and not os.path.exists(path):
        errors.append(f"{label}: no such file or directory: {path}")


def _from_dict(cls, data: Dict[str, Any]):
    """Build a dataclass from a dict, rejecting unknown keys with a clear error."""
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"unknown key(s) in `{cls.__name__}`: {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )
    return cls(**data)


# ----------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------
@dataclass
class DataConfig:
    """Genome, sequence intervals and single-cell inputs.

    Attributes:
        fasta_file: Reference genome FASTA.
        bed_file: Borzoi-style sequence BED with a fold column.
        adata_rna_plus / adata_rna_minus / adata_atac: Per-modality AnnData
            files holding the single-cell fragment coverage. Leave
            ``adata_atac`` unset to train on RNA alone, with
            ``model.num_tracks: 2``.
        embedding: Parquet with an ``embedding`` column of cell-state vectors.
        adata_rna_key: ``obsm`` key holding the RNA fragment matrix, read
            without loading the file into memory.
        adata_atac_key: ``obsm`` key holding the ATAC insertion matrix. Set to
            ``null`` to read the file eagerly with scanpy instead, which is
            only workable for small ATAC files.
        neighbors: ``.npz`` sparse neighbour graph. Optional; without it each
            cell is decoded on its own.
        val_bed_file: Optional separate BED for validation. If unset, the
            validation set is taken from ``bed_file`` using ``val_fold``.
        test_fold / val_fold: Fold labels held out from training. ``-1``
            disables the split.
        context_length: Input sequence length in bp.
        shift_augs: Random shift range (bp) applied during training.
        rc_aug: Whether to apply reverse-complement augmentation.
        clip_soft: Soft-clipping threshold for coverage normalisation.
        cell_sample_size: Cells decoded per sequence during training.
        normalize_atac: Whether to depth-normalise ATAC coverage.
    """

    fasta_file: str
    bed_file: str
    adata_rna_plus: str
    adata_rna_minus: str
    embedding: str
    adata_atac: Optional[str] = None
    neighbors: Optional[str] = None
    val_bed_file: Optional[str] = None
    adata_rna_key: str = "fragment_single"
    adata_atac_key: Optional[str] = "insertion"
    test_fold: int = -1
    val_fold: int = -1
    context_length: int = 524_288
    shift_augs: List[int] = field(default_factory=lambda: [-3, 3])
    rc_aug: bool = True
    clip_soft: float = 5.0
    cell_sample_size: int = 64
    normalize_atac: bool = True

    PATH_KEYS = (
        "fasta_file", "bed_file", "adata_rna_plus", "adata_rna_minus",
        "adata_atac", "embedding", "neighbors", "val_bed_file",
    )

    def resolve(self, root: Path) -> None:
        for key in self.PATH_KEYS:
            setattr(self, key, resolve_path(getattr(self, key), root))

    def validate(self, errors: List[str], check_paths: bool = True) -> None:
        if check_paths:
            for key in self.PATH_KEYS:
                _require_exists(getattr(self, key), f"data.{key}", errors)
        if self.context_length % 128 != 0:
            errors.append(
                f"data.context_length ({self.context_length}) must be a "
                "multiple of 128, AlphaGenome's encoder stride."
            )

    @property
    def rna_only(self) -> bool:
        return self.adata_atac is None

    @property
    def num_tracks(self) -> int:
        """Tracks per cell in the targets: RNA plus and minus, then ATAC."""
        return 2 if self.rna_only else 3

    @property
    def adata_paths(self) -> Dict[str, str]:
        paths = {
            "rna_plus": self.adata_rna_plus,
            "rna_minus": self.adata_rna_minus,
        }
        if not self.rna_only:
            paths["atac"] = self.adata_atac
        return paths


@dataclass
class LoraScope:
    """Which regions of AlphaGenome receive LoRA adapters."""

    encoder_convs: bool = True
    decoder_convs: bool = False
    attn: bool = True
    mlp: bool = True


@dataclass
class ModelConfig:
    """Architecture and fine-tuning scope.

    Attributes:
        cell_emb_dim: Width of the cell-state embedding.
        num_tracks: Output tracks per cell (3 = RNA +/- and ATAC).
        embedding_dim: Sequence embedding width fed to the cell-state conv.
        pretrained_model: AlphaGenome checkpoint (``.safetensors`` or
            ``.pth``). Unset means random initialisation.
        organism_index: 0 for human, 1 for mouse.
        pool_factor: Output resolution in bp when decoding to 1 bp
            (``early_exit_32bp: false``): the 1 bp embedding is mean-pooled by
            this factor. ``1`` predicts at base resolution. Must stay 32 with
            the early exit, which already ends at 32 bp.
        target_bins: Bins kept after the centred crop, at the output
            resolution (6144 at 32 bp, 196608 at 1 bp).
        early_exit_32bp: Leave the decoder at 32 bp instead of decoding to
            1 bp and pooling back.
        freeze_batchnorm: Keep AlphaGenome's normalisation statistics fixed.
        use_lora: Fine-tune with LoRA rather than updating all weights.
        lora: Region flags used to build the adapter target regex.
        lora_targets: Literal regex, overriding ``lora`` when set.
        lora_r / lora_alpha / lora_dropout: Adapter hyperparameters.
        standardized_conv_lora: Apply convolutional adapters inside
            AlphaGenome's weight standardisation. See ``scooby.scag.lora``.
        head_input_norm: Normalise the sequence embedding before the
            cell-state decoder, making its input scale independent of drift
            in the backbone.
    """

    cell_emb_dim: int
    num_tracks: int = 3
    embedding_dim: int = 1536
    pretrained_model: Optional[str] = None
    organism_index: int = 0
    pool_factor: int = 32
    target_bins: Optional[int] = 6144
    early_exit_32bp: bool = True
    freeze_batchnorm: bool = True
    use_transform_emb: bool = False
    use_lora: bool = True
    lora: LoraScope = field(default_factory=LoraScope)
    lora_targets: Optional[str] = None
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    standardized_conv_lora: bool = True
    head_input_norm: bool = False

    def __post_init__(self):
        if isinstance(self.lora, dict):
            self.lora = _from_dict(LoraScope, self.lora)

    def resolve(self, root: Path) -> None:
        self.pretrained_model = resolve_path(self.pretrained_model, root)

    def validate(self, errors: List[str], check_paths: bool = True) -> None:
        if check_paths:
            _require_exists(self.pretrained_model, "model.pretrained_model", errors)
        if self.early_exit_32bp and self.pool_factor != 32:
            errors.append(
                f"model.pool_factor is {self.pool_factor}, but early_exit_32bp "
                "already ends at 32 bp. Set early_exit_32bp: false to change "
                "the output resolution."
            )
        if self.use_lora and self.lora_targets is None and not any(
            (self.lora.encoder_convs, self.lora.decoder_convs,
             self.lora.attn, self.lora.mlp)
        ):
            errors.append(
                "model.lora: every region is disabled, so LoRA has nothing "
                "to adapt. Enable a region or set model.use_lora: false."
            )

    def model_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for ``ScoobyAlphaGenome.__init__``."""
        return dict(
            cell_emb_dim=self.cell_emb_dim,
            embedding_dim=self.embedding_dim,
            n_tracks=self.num_tracks,
            pool_factor=self.pool_factor,
            target_bins=self.target_bins,
            organism_index=self.organism_index,
            early_exit_32bp=self.early_exit_32bp,
            freeze_batchnorm=self.freeze_batchnorm,
            use_transform_emb=self.use_transform_emb,
            head_input_norm=self.head_input_norm,
        )


@dataclass
class TrainingConfig:
    """Optimisation settings.

    Attributes:
        batch_size: Sequences per step per device.
        lr: Learning rate for the backbone and adapters.
        cellstate_lr: Separate learning rate for the cell-state decoder.
        encoder_lora_lr: Optional learning rate for convolutional adapters
            in the encoder. Defaults to ``lr``.
        wd: Weight decay.
        clip_global_norm: Gradient-norm clipping threshold.
        num_epochs: Passes over the sequence set.
        steps_per_epoch: Training sequences per epoch. Divided by
            ``batch_size`` and the process count to size the LR schedule.
        warmup_steps: Learning-rate warm-up, in optimizer steps.
        num_workers: Dataloader workers per process.
        compile: Compile the model with TorchInductor.
        compile_mode: TorchInductor mode. ``reduce-overhead`` uses CUDA
            graphs, which do not combine with DDP.
        ddp_static_graph: Pass ``static_graph`` to DDP.
        eval_every_n / save_every_n: Step intervals for validation and
            checkpointing.
        log_every_n: Steps between logging and collapse checks.
        total_weight: Weight of the Poisson term in the loss.
        loss_epsilon: Added to every predicted and target bin before the loss;
            keep it well below the mean per-bin target.
        collapse_patience: Window of recent steps over which the median
            largest prediction is checked. ``0`` disables the check. See
            ``scooby.scag.train``.
        collapse_threshold: Median largest prediction below which training aborts.
    """

    batch_size: int = 1
    lr: float = 4e-5
    cellstate_lr: float = 1e-4
    encoder_lora_lr: Optional[float] = None
    wd: float = 1e-6
    clip_global_norm: float = 1.0
    warmup_steps: int = 5000
    num_epochs: int = 4
    steps_per_epoch: int = 45_000
    num_workers: int = 8
    compile: bool = False
    compile_mode: str = "default"
    ddp_static_graph: bool = True
    eval_every_n: int = 256
    save_every_n: int = 2000
    log_every_n: int = 100
    total_weight: float = 0.2
    loss_epsilon: float = 1e-6
    collapse_patience: int = 50
    collapse_threshold: float = 1e-8

    COMPILE_MODES = (
        "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs",
    )

    def validate(self, errors: List[str]) -> None:
        if self.log_every_n < 1:
            errors.append("training.log_every_n must be at least 1")
        if self.compile and self.compile_mode not in self.COMPILE_MODES:
            errors.append(
                f"training.compile_mode: unknown mode {self.compile_mode!r}. "
                f"Valid modes: {list(self.COMPILE_MODES)}"
            )
        # This bounds warm-up by the single-process step count. The real
        # check needs the world size and happens in `scooby.scag.train`.
        if self.warmup_steps >= self.steps_per_epoch * self.num_epochs:
            errors.append(
                "training.warmup_steps exceeds the total number of steps; "
                "the learning rate would never leave warm-up."
            )


@dataclass
class ScAGConfig:
    """Top-level config, normally built by :meth:`from_yaml`.

    ``data`` is required for training and optional otherwise. Configs stored
    alongside a checkpoint for inference, and those recovered from older runs,
    may omit it.
    """

    output_dir: str
    run_name: str
    model: ModelConfig
    data: Optional[DataConfig] = None
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(
        cls, path: str | os.PathLike, check_paths: bool = True
    ) -> "ScAGConfig":
        """Load, resolve and validate a config file.

        Args:
            path: The YAML file.
            check_paths: Require every declared file to exist. Turn this off
                to read a config only to rebuild a model, for example the
                ``scag_config.yaml`` stored next to a checkpoint, whose
                training data and AlphaGenome weights need not be present.

        Raises:
            ValueError: if keys are unknown or declared paths are missing.
                All problems are reported at once.
        """
        path = Path(path).expanduser().resolve()
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        root = path.parent

        for section, klass in (("data", DataConfig), ("model", ModelConfig),
                               ("training", TrainingConfig)):
            if section in raw and isinstance(raw[section], dict):
                raw[section] = _from_dict(klass, raw[section])
        cfg = _from_dict(cls, raw)
        cfg.resolve(root)
        cfg.validate(check_paths=check_paths)
        return cfg

    def resolve(self, root: Path) -> None:
        self.output_dir = resolve_path(self.output_dir, root)
        self.model.resolve(root)
        if self.data is not None:
            self.data.resolve(root)

    def validate(self, check_paths: bool = True) -> None:
        errors: List[str] = []
        if self.data is not None:
            self.data.validate(errors, check_paths=check_paths)
            if self.model.num_tracks != self.data.num_tracks:
                errors.append(
                    f"model.num_tracks is {self.model.num_tracks} but the data "
                    f"yield {self.data.num_tracks} tracks per cell "
                    f"({'RNA only' if self.data.rna_only else 'RNA and ATAC'}). "
                    "Use 2 without data.adata_atac and 3 with it."
                )
        self.model.validate(errors, check_paths=check_paths)
        self.training.validate(errors)
        if errors:
            raise ValueError(
                "invalid scAG config:\n  - " + "\n  - ".join(errors)
            )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if data.get("data") is None:
            data.pop("data", None)
        return data

    def require_data(self) -> DataConfig:
        """Return the data config, or explain that the run needs one."""
        if self.data is None:
            raise ValueError(
                "this config has no `data` section, so it can only be used to "
                "rebuild a model, not to train one"
            )
        return self.data

    def dump(self, path: str | os.PathLike) -> None:
        """Write the resolved config next to a checkpoint."""
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @property
    def checkpoint_root(self) -> str:
        return os.path.join(self.output_dir, self.run_name)
