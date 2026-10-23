"""Dataset construction from a :class:`~scooby.scag.config.ScAGConfig`.

Thin wrappers around scooby's dataloaders. Their job is to keep every path
in the config file rather than in the training script.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
import scipy.sparse
from torch.utils.data import Dataset
from enformer_pytorch.data import GenomeIntervalDataset

from scooby.data import onTheFlyDataset, onTheFlyMultiomeDataset
from scooby.utils.utils import read_backed

from .config import DataConfig

FOLD_COLUMN = "column_4"


def _cache_anndata_version_lookup() -> None:
    """Stop anndata re-reading scipy's package metadata on every backed read.

    With scipy >= 1.17, anndata 0.12's ``validate_indices`` calls
    ``importlib.metadata.version("scipy")`` on each row lookup, which parses
    the installed metadata from disk. scooby's dataset reads one row per cell
    and strand, so at 512 cells this was ~3 ms x 1024 lookups per sample, most
    of the time spent building it. The installed version cannot change while
    the process runs, so caching it is safe.
    """
    try:
        from anndata._core import sparse_dataset
    except ImportError:
        return
    lookup = getattr(sparse_dataset, "version", None)
    if lookup is not None and not hasattr(lookup, "cache_info"):
        sparse_dataset.version = functools.lru_cache(maxsize=None)(lookup)


_cache_anndata_version_lookup()


def load_adatas(cfg: DataConfig) -> dict:
    """Open the per-modality AnnData files named in the config.

    The coverage matrices live in ``obsm`` and are read backed, so files that
    do not fit in memory are never materialised. The NeurIPS ATAC file is
    15 GB, which rules out an eager read. An RNA-only config has no ATAC
    entry.
    """
    paths = cfg.adata_paths
    adatas = {
        "rna_plus": read_backed(h5py.File(paths["rna_plus"]), cfg.adata_rna_key),
        "rna_minus": read_backed(h5py.File(paths["rna_minus"]), cfg.adata_rna_key),
    }
    if cfg.rna_only:
        return adatas
    if cfg.adata_atac_key is None:
        adatas["atac"] = sc.read(paths["atac"])
    else:
        adatas["atac"] = read_backed(h5py.File(paths["atac"]), cfg.adata_atac_key)
    return adatas


def load_cell_state(cfg: DataConfig):
    """Load the cell-state embedding table and the neighbour graph."""
    embedding = pd.read_parquet(cfg.embedding)
    neighbors = (
        scipy.sparse.load_npz(cfg.neighbors) if cfg.neighbors is not None else None
    )
    return embedding, neighbors


def embedding_dim(embedding: pd.DataFrame) -> int:
    """Width of the stored cell-state vectors."""
    return int(np.asarray(embedding["embedding"].iloc[0]).shape[0])


def sequence_dataset(
    cfg: DataConfig, split: str = "train", bed_file: Optional[str] = None
) -> GenomeIntervalDataset:
    """Build the interval dataset for one split.

    Args:
        cfg: The data config.
        split: ``"train"`` applies the configured augmentations and excludes
            the validation and test folds; ``"val"`` selects the validation
            fold without augmentation.
        bed_file: Override the BED file, for example to score a custom set of
            regions.

    Returns:
        A ``GenomeIntervalDataset`` over the requested split.
    """
    if split == "train":
        filter_fn = lambda df: df.filter(  # noqa: E731
            (pl.col(FOLD_COLUMN) != f"fold{cfg.test_fold}")
            & (pl.col(FOLD_COLUMN) != f"fold{cfg.val_fold}")
        )
        shift_augs, rc_aug = tuple(cfg.shift_augs), cfg.rc_aug
        bed = bed_file or cfg.bed_file
    elif split == "val":
        bed = bed_file or cfg.val_bed_file or cfg.bed_file
        if cfg.val_bed_file is not None and bed_file is None:
            filter_fn = lambda df: df.filter(pl.col("column_2") >= 0)  # noqa: E731
        else:
            filter_fn = lambda df: df.filter(  # noqa: E731
                pl.col(FOLD_COLUMN) == f"fold{cfg.val_fold}"
            )
        shift_augs, rc_aug = (0, 0), False
    else:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    return GenomeIntervalDataset(
        bed_file=bed,
        fasta_file=cfg.fasta_file,
        filter_df_fn=filter_fn,
        return_seq_indices=False,
        shift_augs=shift_augs,
        rc_aug=rc_aug,
        return_augs=True,
        context_length=cfg.context_length,
        chr_bed_to_fasta_map={},
    )


def multiome_dataset(
    cfg: DataConfig,
    adatas: dict,
    embedding: pd.DataFrame,
    neighbors,
    ds: GenomeIntervalDataset,
    cell_sample_size: Optional[int] = None,
) -> Union[onTheFlyDataset, onTheFlyMultiomeDataset]:
    """Pair an interval dataset with the single-cell coverage targets.

    Without ATAC this is scooby's RNA-only dataset, which yields the plus and
    minus tracks per cell in the same layout as the first two multiome tracks.
    """
    if cfg.rna_only:
        return onTheFlyDataset(
            adatas["rna_plus"],
            adatas["rna_minus"],
            embedding=embedding,
            ds=ds,
            clip_soft=cfg.clip_soft,
            neighbors=neighbors,
            cell_sample_size=cell_sample_size or cfg.cell_sample_size,
            cell_weights=None,
        )
    return onTheFlyMultiomeDataset(
        adatas=adatas,
        neighbors=neighbors,
        ds=ds,
        cell_sample_size=cell_sample_size or cfg.cell_sample_size,
        cell_weights=None,
        normalize_atac=cfg.normalize_atac,
        clip_soft=cfg.clip_soft,
        embedding=embedding,
    )


def build_datasets(
    cfg: DataConfig,
) -> Tuple[Dataset, Optional[Dataset], int]:
    """Build the training and validation datasets and the embedding width.

    Returns:
        ``(train_dataset, val_dataset, cell_emb_dim)``. The validation dataset
        is ``None`` when the configured fold selects no sequences.
    """
    adatas = load_adatas(cfg)
    embedding, neighbors = load_cell_state(cfg)

    train_ds = sequence_dataset(cfg, "train")
    train = multiome_dataset(cfg, adatas, embedding, neighbors, train_ds)

    val_ds = sequence_dataset(cfg, "val")
    val = (
        multiome_dataset(
            cfg, adatas, embedding, neighbors, val_ds, cell_sample_size=32
        )
        if len(val_ds) > 0
        else None
    )
    return train, val, embedding_dim(embedding)
