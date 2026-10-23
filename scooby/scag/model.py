"""Scooby with an AlphaGenome sequence backbone.

``ScoobyAlphaGenome`` is a drop-in counterpart of ``scooby.modeling.Scooby``
that replaces Borzoi with AlphaGenome. Sequence embeddings are taken at 32 bp
resolution and decoded per cell by the same cell-state convolution, so the
downstream machinery is unchanged.
"""

from __future__ import annotations

import os
import re
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.embeddings import OutputEmbedder
from alphagenome_pytorch.layers import RMSBatchNorm

batch_conv = torch.vmap(F.conv1d, chunk_size=1024)

# AlphaGenome's own prediction heads. Scooby replaces the head entirely, so
# they are dropped to save memory and to keep them out of LoRA's reach.
_AG_HEAD_ATTRS = (
    "heads",
    "contact_maps_head",
    "splice_sites_classification_head",
    "splice_sites_usage_head",
    "splice_sites_junction_head",
)

# Channel count of AlphaGenome's decoder at 32 bp, used by the early-exit path.
_DECODER_32BP_CHANNELS = 1408

CONFIG_FILENAME = "scag_config.yaml"
WEIGHTS_FILENAME = "model.safetensors"
LEGACY_CONFIG_FILENAME = "training_config.yaml"


def load_ag_state_dict(path: str) -> dict:
    """Load an AlphaGenome state dict from ``.pth`` or ``.safetensors``."""
    if os.path.splitext(path)[1].lower() == ".safetensors":
        from safetensors.torch import load_file

        return load_file(path, device="cpu")
    state = torch.load(path, weights_only=True, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def unwrap(model: nn.Module) -> "ScoobyAlphaGenome":
    """Return the underlying model from a peft wrapper.

    peft proxies only ``forward``, so any other method call has to go through
    the wrapped module.
    """
    inner = model
    for _ in range(4):
        if isinstance(inner, ScoobyAlphaGenome):
            return inner
        if hasattr(inner, "base_model"):
            inner = inner.base_model
        elif hasattr(inner, "model"):
            inner = inner.model
        else:
            break
    if not isinstance(inner, ScoobyAlphaGenome):
        raise TypeError(f"could not reach a ScoobyAlphaGenome inside {type(model)}")
    return inner


class ScoobyAlphaGenome(nn.Module):
    """Scooby with AlphaGenome as the sequence backbone.

    Args:
        cell_emb_dim: Width of the cell-state embedding.
        embedding_dim: Width of the sequence embedding fed to the cell-state
            convolution.
        n_tracks: Output tracks per cell, 2 for stranded RNA or 3 with ATAC.
        pool_factor: Mean-pooling factor from 1 bp to bin resolution. Ignored
            when ``early_exit_32bp`` is set, where the output is already at
            32 bp.
        target_bins: Bins kept after a centred crop. 6144 matches Borzoi's
            centre-bin output and the multiome targets. ``None`` keeps all
            bins, which is useful for tests on short sequences.
        organism_index: 0 for human, 1 for mouse. Used when ``forward`` is
            called without an explicit index.
        alphagenome: A pre-built backbone. If omitted, one is constructed.
        ag_pretrained_path: AlphaGenome weights to load into a fresh
            backbone. Ignored when ``alphagenome`` is given, and unnecessary
            when restoring a scAG checkpoint, which already holds them.
        num_organisms: Passed to ``AlphaGenome`` when building a backbone.
        disable_cache: Skip caching sequence embeddings between calls.
        use_transform_emb: Add a zero-initialised residual 1x1 convolution
            stack on the pooled embedding.
        cachesize: Number of cached ``(sequence, embedding)`` pairs.
        drop_unused_heads: Delete AlphaGenome's prediction heads.
        early_exit_32bp: Stop the decoder after ``up_blocks[1]``, which is
            already at 32 bp, and embed from there. This skips the seven-level
            decoder pyramid and the 1 bp embedder. The fresh 32 bp embedder is
            randomly initialised, so checkpoints are not interchangeable with
            the full-resolution path.
        freeze_batchnorm: Pin AlphaGenome's normalisation layers to eval mode,
            make ``train()`` a no-op on them, and freeze their affine
            parameters.
        head_input_norm: Normalise the sequence embedding over channels before
            the cell-state convolution. This makes the decoder's input scale
            independent of drift in the backbone during fine-tuning. Off by
            default because it changes the architecture.
    """

    def __init__(
        self,
        cell_emb_dim: int,
        embedding_dim: int = 1536,
        n_tracks: int = 2,
        pool_factor: int = 32,
        target_bins: Optional[int] = 6144,
        organism_index: int = 0,
        alphagenome: Optional[AlphaGenome] = None,
        ag_pretrained_path: Optional[str] = None,
        num_organisms: int = 2,
        disable_cache: bool = True,
        use_transform_emb: bool = False,
        cachesize: int = 2,
        drop_unused_heads: bool = True,
        early_exit_32bp: bool = False,
        freeze_batchnorm: bool = True,
        head_input_norm: bool = False,
    ):
        super().__init__()

        if alphagenome is None:
            alphagenome = AlphaGenome(num_organisms=num_organisms)
            if ag_pretrained_path is not None:
                alphagenome.load_state_dict(load_ag_state_dict(ag_pretrained_path))
        self.alphagenome = alphagenome

        if drop_unused_heads:
            self._strip_ag_heads()

        self.cell_emb_dim = cell_emb_dim
        self.embedding_dim = embedding_dim
        self.n_tracks = n_tracks
        self.pool_factor = pool_factor
        self.target_bins = target_bins
        self.organism_index = organism_index
        self.disable_cache = disable_cache
        self.cachesize = cachesize
        self.use_transform_emb = use_transform_emb
        self.early_exit_32bp = early_exit_32bp
        self.freeze_batchnorm = freeze_batchnorm

        if early_exit_32bp:
            self._enable_early_exit_32bp()

        # AlphaGenome ships no dropout, but keep Scooby's habit of pinning any
        # dropout in the backbone to eval mode.
        for module in self.alphagenome.modules():
            if isinstance(module, nn.Dropout):
                module.eval()

        if freeze_batchnorm:
            self._freeze_norm_layers()

        self.avgpool = nn.AvgPool1d(kernel_size=pool_factor, stride=pool_factor)

        self.head_input_norm = (
            nn.GroupNorm(1, embedding_dim) if head_input_norm else None
        )

        self.cell_state_to_conv = nn.Sequential(
            nn.Linear(cell_emb_dim, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, (self.embedding_dim + 1) * self.n_tracks),
        )
        nn.init.zeros_(self.cell_state_to_conv[-1].bias)

        if use_transform_emb:
            self.transform_emb = nn.Sequential(
                nn.Conv1d(embedding_dim, 4096, 1),
                nn.GELU(),
                nn.Conv1d(4096, embedding_dim, 1),
                nn.GELU(),
            )
            nn.init.zeros_(self.transform_emb[-2].weight)
            nn.init.zeros_(self.transform_emb[-2].bias)

        self.sequences, self.last_embs = [], []

    # ------------------------------------------------------------------
    # Backbone surgery
    # ------------------------------------------------------------------
    def _strip_ag_heads(self) -> None:
        for attr in _AG_HEAD_ATTRS:
            if hasattr(self.alphagenome, attr):
                delattr(self.alphagenome, attr)

    def _freeze_norm_layers(self) -> None:
        """Freeze every normalisation layer in the backbone.

        ``RMSBatchNorm`` rescales by a stored running variance and is
        mode-independent, but ``train()`` is overridden anyway so that
        accelerate or a ``SyncBatchNorm`` conversion cannot re-enable
        statistics updates in any layer.
        """
        for module in self.alphagenome.modules():
            if not isinstance(module, (nn.BatchNorm1d, RMSBatchNorm)):
                continue
            module.eval()

            def _no_op_train(self, mode=True):
                return self

            module.train = _no_op_train.__get__(module, module.__class__)
            for p in module.parameters(recurse=False):
                p.requires_grad = False

    def _enable_early_exit_32bp(self) -> None:
        """Trim the decoder to 32 bp and add a matching output embedder.

        Decoding stops after ``up_blocks[1]``, which outputs 1408 channels at
        S/32. A fresh ``OutputEmbedder`` maps those to ``embedding_dim``, with
        a skip from the 128 bp branch mirroring AlphaGenome's ``embedder_1bp``.
        The high-resolution ``up_blocks[2:]``, ``embedder_1bp`` and
        ``embedder_pair`` are deleted.
        """
        ag = self.alphagenome

        ag.decoder.up_blocks = nn.ModuleList(list(ag.decoder.up_blocks)[:2])
        ag.decoder.bin_sizes = ag.decoder.bin_sizes[:2]

        for attr in ("embedder_1bp", "embedder_pair"):
            if hasattr(ag, attr):
                delattr(ag, attr)

        embedder = OutputEmbedder(
            in_channels=_DECODER_32BP_CHANNELS,
            out_channels=self.embedding_dim,
            num_organisms=ag.embedder_128bp.num_organisms,
        )
        embedder.project_skip = nn.Conv1d(
            ag.embedder_128bp.out_channels,
            self.embedding_dim,
            kernel_size=1,
            bias=False,
        )
        self.embedder_32bp = embedder

    # ------------------------------------------------------------------
    # Sequence path
    # ------------------------------------------------------------------
    def _forward_ag_to_32bp_emb(self, sequence, organism_idx):
        """Run the backbone up to a 32 bp embedding via the early-exit path."""
        ag = self.alphagenome
        compute_dtype = ag.dtype_policy.compute_dtype
        dna = ag.dtype_policy.cast_to_compute(sequence)

        trunk, intermediates = ag.encoder(dna)  # (B, 1536, S/128) NCL

        trunk = trunk.transpose(1, 2)
        trunk = trunk + ag.organism_embed(organism_idx).unsqueeze(1)
        # The pair representation is needed for the tower's attention bias
        # but not downstream.
        trunk, _pair = ag.tower(trunk, compute_dtype=compute_dtype)
        trunk_ncl = trunk.transpose(1, 2)

        emb_128 = ag.embedder_128bp(trunk_ncl, organism_idx, channels_last=False)

        x = trunk_ncl
        for i, bin_size in enumerate(ag.decoder.bin_sizes):
            x = ag.decoder.up_blocks[i](x, intermediates[f"bin_size_{bin_size}"])
        return self.embedder_32bp(x, organism_idx, skip_x=emb_128, channels_last=False)

    def _center_crop_bins(self, x):
        if self.target_bins is None:
            return x
        length = x.size(-1)
        if length == self.target_bins:
            return x
        if length < self.target_bins:
            raise ValueError(
                f"sequence yields {length} bins after pooling but target_bins "
                f"is {self.target_bins}; lengthen the input or lower target_bins"
            )
        start = (length - self.target_bins) // 2
        return x[..., start : start + self.target_bins]

    def _organism_idx_for(self, sequence, organism_index):
        if organism_index is None:
            organism_index = self.organism_index
        if isinstance(organism_index, int):
            return torch.full(
                (sequence.size(0),),
                organism_index,
                device=sequence.device,
                dtype=torch.long,
            )
        return organism_index.to(device=sequence.device, dtype=torch.long)

    def forward_seq_to_emb(self, sequence, organism_index=None):
        """Embed a sequence at bin resolution.

        Args:
            sequence: One-hot DNA in NLC layout ``(B, S, 4)``. ``S`` must be
                divisible by 128 and by ``pool_factor``.
            organism_index: Optional per-sample organism index ``(B,)``.

        Returns:
            ``(B, embedding_dim, bins)``, ready for the cell-state convolution.
        """
        organism_idx = self._organism_idx_for(sequence, organism_index)
        if self.early_exit_32bp:
            x = self._forward_ag_to_32bp_emb(sequence, organism_idx)
        else:
            out = self.alphagenome.encode(
                sequence, organism_idx, resolutions=(1,), channels_last=False
            )
            x = self.avgpool(out["embeddings_1bp"])
        x = self._center_crop_bins(x)

        if self.use_transform_emb:
            x = x + self.transform_emb(x)
        if self.head_input_norm is not None:
            x = self.head_input_norm(x)

        if not self.training and not self.disable_cache:
            if len(self.sequences) == self.cachesize:
                self.sequences, self.last_embs = [], []
            self.sequences.append(sequence)
            self.last_embs.append(x)
        return x

    # ------------------------------------------------------------------
    # Cell-state path
    # ------------------------------------------------------------------
    def forward_cell_embs_only(self, cell_emb):
        """Turn cell-state embeddings into per-cell convolution kernels."""
        bs, no_cell_embs, _ = cell_emb.shape
        cell_emb = rearrange(cell_emb, "b n d -> (b n) d")
        w = self.cell_state_to_conv(cell_emb)
        w = rearrange(w, "(b n) (l d) -> b (n l) d", b=bs, l=self.n_tracks)
        biases = w[:, :, -1:].reshape(bs, no_cell_embs * self.n_tracks)
        weights = w[:, :, :-1].reshape(
            bs, no_cell_embs * self.n_tracks, self.embedding_dim, 1
        )
        return weights, biases

    def forward_convs_on_emb(
        self, seq_emb, cell_emb_conv_weights, cell_emb_conv_biases, bins_to_predict=None
    ):
        x = seq_emb if bins_to_predict is None else seq_emb[:, :, bins_to_predict]
        out = batch_conv(x, cell_emb_conv_weights, cell_emb_conv_biases)
        return F.softplus(out).permute(0, 2, 1)

    def forward_sequence_w_convs(
        self,
        sequence,
        cell_emb_conv_weights,
        cell_emb_conv_biases,
        bins_to_predict=None,
        organism_index=None,
    ):
        """Embed ``sequence`` and decode it with per-cell kernels."""
        if self.sequences and not self.training and not self.disable_cache:
            for i, cached in enumerate(self.sequences):
                if torch.equal(sequence, cached):
                    emb = self.last_embs[i]
                    return self.forward_convs_on_emb(
                        emb,
                        cell_emb_conv_weights.to(emb.dtype),
                        cell_emb_conv_biases.to(emb.dtype),
                        bins_to_predict,
                    )

        x = self.forward_seq_to_emb(sequence, organism_index=organism_index)
        return self.forward_convs_on_emb(
            x,
            cell_emb_conv_weights.to(x.dtype),
            cell_emb_conv_biases.to(x.dtype),
            bins_to_predict,
        )

    def forward(self, sequence, cell_emb, organism_index=None):
        """Predict per-cell profiles.

        Args:
            sequence: One-hot DNA in NLC layout ``(B, S, 4)``. Unlike the
                Borzoi-based Scooby, no permutation to NCL is needed.
            cell_emb: ``(B, num_cells, cell_emb_dim)`` cell-state embeddings.
            organism_index: Optional ``(B,)`` organism indices.

        Returns:
            ``(B, bins, num_cells * n_tracks)`` predicted coverage.
        """
        weights, biases = self.forward_cell_embs_only(cell_emb)
        return self.forward_sequence_w_convs(
            sequence, weights, biases, organism_index=organism_index
        )

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    def save_pretrained(self, save_directory: str, config=None, **kwargs) -> None:
        """Write weights and, if given, the config needed to rebuild the model.

        Args:
            save_directory: Destination directory, created if absent.
            config: A :class:`~scooby.scag.config.ScAGConfig`. Without it,
                :meth:`from_pretrained` needs explicit architecture arguments.
        """
        from safetensors.torch import save_file

        os.makedirs(save_directory, exist_ok=True)
        if config is not None:
            config.dump(os.path.join(save_directory, CONFIG_FILENAME))
        state = {k: v.contiguous() for k, v in self.state_dict().items()}
        save_file(state, os.path.join(save_directory, WEIGHTS_FILENAME))

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: Union[str, os.PathLike],
        config=None,
        device: Optional[Union[str, torch.device]] = None,
        train: bool = False,
        strict: bool = True,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        **overrides,
    ):
        """Rebuild a trained model from a checkpoint directory.

        The architecture and LoRA scope are read from ``scag_config.yaml``
        written at save time, or from the ``training_config.yaml`` left by
        older runs. Either can be overridden by passing ``config`` directly.
        Paths in the stored config (training data, AlphaGenome weights) are
        not needed to rebuild the model and are not checked.

        Args:
            checkpoint_dir: Directory holding the weights and config. If it
                is not an existing directory, it is taken as a Hugging Face
                Hub repo id, for example ``"johahi/neurips-scooby-2"``, and
                downloaded with ``huggingface_hub.snapshot_download``.
            config: A :class:`~scooby.scag.config.ScAGConfig` to use instead of the
                one on disk.
            device: Device to move the model to.
            train: Whether to return the model ready for further training.
            strict: Whether to require an exact match between checkpoint and
                model parameters.
            revision: Hub only: branch, tag or commit to download.
            cache_dir: Hub only: where to cache the download.
            token: Hub only: access token for a private repo.
            **overrides: Fields to override on the loaded ``model`` config,
                for example ``cell_emb_dim=32``.

        Returns:
            The model, wrapped by peft when the run used LoRA. Use
            :func:`unwrap` to reach the ``ScoobyAlphaGenome`` itself.

        Raises:
            FileNotFoundError: if no weights or no usable config is found.
        """
        from .config import ScAGConfig

        checkpoint_dir = resolve_checkpoint(
            checkpoint_dir, revision=revision, cache_dir=cache_dir, token=token
        )
        weights_path = _find_weights(checkpoint_dir)
        if config is None:
            config = _load_checkpoint_config(checkpoint_dir)
        if not isinstance(config, ScAGConfig):
            raise TypeError("config must be a ScAGConfig")

        model_cfg = config.model
        for key, value in overrides.items():
            if not hasattr(model_cfg, key):
                raise ValueError(f"unknown model config field: {key}")
            setattr(model_cfg, key, value)

        kwargs = model_cfg.model_kwargs()
        # The checkpoint carries the backbone weights, so skip the pretrained
        # AlphaGenome file even when the training config named one.
        model = cls(ag_pretrained_path=None, disable_cache=not train, **kwargs)

        if model_cfg.use_lora:
            from .lora import apply_lora

            model = apply_lora(model, model_cfg, train=train)

        _load_weights(model, weights_path, strict=strict)
        if device is not None:
            model = model.to(device)
        model.train(train)
        return model


# A Hub repo id is ``name`` or ``namespace/name``; anything else that is not a
# local directory is reported as a missing path rather than sent to the Hub.
_REPO_ID = re.compile(r"^(?!\.)[\w.-]+(/(?!\.)[\w.-]+)?$")


def resolve_checkpoint(
    checkpoint: Union[str, os.PathLike],
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    """Return a local checkpoint directory, downloading it from the Hub if needed.

    An existing local directory is returned as is. Otherwise ``checkpoint`` is
    treated as a Hugging Face Hub repo id and only the files needed to rebuild
    the model are downloaded.

    Raises:
        FileNotFoundError: if ``checkpoint`` is neither a local directory nor
            a repo id that can be downloaded.
    """
    path = os.path.expanduser(os.fspath(checkpoint))
    if os.path.isdir(path):
        return path
    if os.path.exists(path):
        raise NotADirectoryError(
            f"{checkpoint} is a file; pass the directory holding "
            f"{WEIGHTS_FILENAME} and {CONFIG_FILENAME}"
        )
    if not _REPO_ID.match(path):
        raise FileNotFoundError(f"no such checkpoint directory: {checkpoint}")

    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(
            repo_id=path,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            allow_patterns=[
                WEIGHTS_FILENAME, "pytorch_model.bin",
                CONFIG_FILENAME, LEGACY_CONFIG_FILENAME,
            ],
        )
    except Exception as exc:
        raise FileNotFoundError(
            f"{checkpoint} is not a local directory, and downloading it from "
            f"the Hugging Face Hub failed: {exc}"
        ) from exc


def _find_weights(checkpoint_dir: str) -> str:
    for name in (WEIGHTS_FILENAME, "pytorch_model.bin"):
        path = os.path.join(checkpoint_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"no {WEIGHTS_FILENAME} in {checkpoint_dir}. Note that "
        "adapter_model.safetensors holds only the LoRA weights and cannot "
        "rebuild the model on its own."
    )


PEFT_PREFIX = "base_model.model."
COMPILE_PREFIX = "_orig_mod."

_MISMATCH_HINT = (
    "Check model.cell_emb_dim, model.early_exit_32bp, "
    "model.standardized_conv_lora and the LoRA scope, or pass strict=False."
)


def _strip_compile_prefix(state: dict) -> dict:
    """Drop the ``_orig_mod.`` prefix ``torch.compile`` adds to state dicts.

    `scooby.scag.train` unwraps compilation before saving, so this only matters for
    checkpoints written by other code.
    """
    if not any(COMPILE_PREFIX in k for k in state):
        return state
    return {k.replace(COMPILE_PREFIX, ""): v for k, v in state.items()}


def _match_prefix(state: dict, model: nn.Module) -> dict:
    """Add or strip peft's ``base_model.model.`` prefix to match the model.

    A checkpoint saved from the wrapped model carries the prefix and one saved
    from the unwrapped model does not, so either can be loaded into either.
    """
    state = _strip_compile_prefix(state)
    model_prefixed = any(k.startswith(PEFT_PREFIX) for k in model.state_dict())
    state_prefixed = any(k.startswith(PEFT_PREFIX) for k in state)
    if model_prefixed == state_prefixed:
        return state
    if model_prefixed:
        return {PEFT_PREFIX + k: v for k, v in state.items()}
    return {
        k[len(PEFT_PREFIX):] if k.startswith(PEFT_PREFIX) else k: v
        for k, v in state.items()
    }


def _load_weights(model: nn.Module, path: str, strict: bool = True) -> None:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(path, device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    state = _match_prefix(state, model)
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except RuntimeError as exc:  # shape mismatches surface here
        raise RuntimeError(f"checkpoint does not match the model built from "
                           f"the config.\n{exc}\n{_MISMATCH_HINT}") from exc
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"checkpoint does not match the model built from the config.\n"
            f"  missing ({len(missing)}): {sorted(missing)[:8]}\n"
            f"  unexpected ({len(unexpected)}): {sorted(unexpected)[:8]}\n"
            f"{_MISMATCH_HINT}"
        )


def _load_checkpoint_config(checkpoint_dir: str):
    """Read a checkpoint's config, accepting the pre-``scag`` layout."""
    from .config import ScAGConfig

    path = os.path.join(checkpoint_dir, CONFIG_FILENAME)
    if os.path.exists(path):
        # Data and weight paths in the stored config are not needed here.
        return ScAGConfig.from_yaml(path, check_paths=False)

    legacy = os.path.join(checkpoint_dir, LEGACY_CONFIG_FILENAME)
    if os.path.exists(legacy):
        return legacy_config(legacy)

    raise FileNotFoundError(
        f"neither {CONFIG_FILENAME} nor {LEGACY_CONFIG_FILENAME} found in "
        f"{checkpoint_dir}; pass config= explicitly"
    )


def legacy_config(path: str):
    """Build a config from a ``training_config.yaml`` of an older run.

    Those runs adapted convolutions through peft, so
    ``standardized_conv_lora`` is forced off to keep parameter names aligned
    with the checkpoint.
    """
    import yaml

    from .config import LoraScope, ModelConfig, ScAGConfig, TrainingConfig

    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    m = raw.get("model", {})

    scope = m.get("lora")
    lora = LoraScope(**scope) if isinstance(scope, dict) else LoraScope()
    model_cfg = ModelConfig(
        cell_emb_dim=m["cell_emb_dim"],
        num_tracks=m.get("num_tracks", 3),
        embedding_dim=m.get("embedding_dim", 1536),
        pretrained_model=None,
        organism_index=m.get("organism_index", 0),
        target_bins=m.get("target_bins", 6144),
        early_exit_32bp=m.get("early_exit_32bp", False),
        freeze_batchnorm=m.get("freeze_batchnorm", True),
        use_lora=m.get("use_lora", True),
        lora=lora,
        lora_targets=m.get("lora_targets") or m.get("_resolved_lora_targets"),
        standardized_conv_lora=False,
    )
    # The data section is not needed to rebuild a model.
    return ScAGConfig(
        output_dir=os.path.dirname(path),
        run_name=raw.get("run_name", "legacy"),
        model=model_cfg,
        training=TrainingConfig(),
    )
