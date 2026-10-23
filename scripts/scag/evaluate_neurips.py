#!/usr/bin/env python3
"""Run NeurIPS scAG pseudobulk inference and save reloadable results.

Usage (with scooby installed):

    python scripts/scag/evaluate_neurips.py --checkpoint path/to/checkpoint \
        --submission-data path/to/submission_data

Progress is saved atomically and resumed automatically. The final output
directory contains labelled prediction/target Parquets, cell-type
correlations, metadata, and the resumable NPZ state.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence


# scripts/scag/, which holds configs/ and the default results/ directory.
SCAG_ROOT = Path(__file__).resolve().parent

import numpy as np
import pandas as pd
import polars as pl
import scipy.stats
import torch
import yaml
from enformer_pytorch.data import GenomeIntervalDataset
from tqdm.auto import tqdm

from scooby.scag import (
    ModelConfig,
    ScAGConfig,
    ScoobyAlphaGenome,
    TrainingConfig,
    predict_with_rc,
    unwrap,
)
from scooby.data import onTheFlyPseudobulkDataset
from scooby.utils.transcriptome import Transcriptome
from scooby.utils.utils import (
    get_gene_slice_and_strand,
    get_pseudobulk_count_pred,
    undo_squashed_scale,
)


DEFAULT_OUTPUT_DIR = SCAG_ROOT / "results" / "neurips_inference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained NeurIPS scAG checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--submission-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-genes",
        type=int,
        default=None,
        help="Stop after this many source rows; useful for a smoke test.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Atomically save resumable progress every N source rows.",
    )
    parser.add_argument(
        "--cell-chunk-size",
        type=int,
        default=70_000,
        help="Maximum output tracks decoded together for one cell type.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore existing progress in output-dir and start at row zero.",
    )
    args = parser.parse_args()
    if args.max_genes is not None and args.max_genes < 1:
        parser.error("--max-genes must be positive")
    if args.save_every < 1:
        parser.error("--save-every must be positive")
    if args.cell_chunk_size < 3:
        parser.error("--cell-chunk-size must be at least 3")
    return args


def require_paths(paths: dict[str, Path]) -> None:
    missing = {name: str(path) for name, path in paths.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(f"missing required paths: {missing}")


def inference_config(checkpoint: Path) -> ScAGConfig:
    """Read a checkpoint config without validating training-only AnnData paths."""
    with open(checkpoint / "scag_config.yaml") as handle:
        saved = yaml.safe_load(handle)
    return ScAGConfig(
        output_dir=saved["output_dir"],
        run_name=saved["run_name"],
        model=ModelConfig(**saved["model"]),
        training=TrainingConfig(**saved.get("training", {})),
    )


def normalize_cell_types(cell_type_index: pd.DataFrame) -> pd.DataFrame:
    result = cell_type_index.copy()
    result["size"] = result["cellindex"].map(len)
    result["celltype"] = (
        result["celltype"]
        .str.replace(" ", "_", regex=False)
        .replace({"G/M_prog": "G+M_prog", "MK/E_prog": "MK+E_prog"})
    )
    return result.sort_values("celltype").reset_index(drop=True)


def build_cell_kernels(
    inner,
    embedding: pd.DataFrame,
    cell_type_index: pd.DataFrame,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Decode every cell state once and retain compact fp16 kernels on GPU."""
    kernels = []
    with torch.inference_mode():
        for row in tqdm(
            cell_type_index.itertuples(index=False),
            total=len(cell_type_index),
            desc="Cell types",
        ):
            cell_embs = torch.from_numpy(
                np.vstack(embedding.iloc[row.cellindex]["embedding"].values)
            ).unsqueeze(0).to(device)
            weights, biases = inner.forward_cell_embs_only(cell_embs)
            kernels.append((weights.half(), biases.half()))
            del cell_embs, weights, biases
    return kernels


def progress_path(output_dir: Path) -> Path:
    return output_dir / "progress.npz"


def save_progress(
    output_dir: Path,
    next_index: int,
    source_indices: Sequence[int],
    genes: Sequence[str],
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    cell_types: Sequence[str],
) -> None:
    """Atomically write the small resumable state file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_cell_types = len(cell_types)
    prediction_array = (
        np.stack(predictions).astype(np.float32, copy=False)
        if predictions
        else np.empty((0, n_cell_types), dtype=np.float32)
    )
    target_array = (
        np.stack(targets).astype(np.float32, copy=False)
        if targets
        else np.empty((0, n_cell_types), dtype=np.float32)
    )
    temporary = output_dir / ".progress.tmp.npz"
    np.savez_compressed(
        temporary,
        next_index=np.asarray(next_index, dtype=np.int64),
        source_indices=np.asarray(source_indices, dtype=np.int64),
        genes=np.asarray(genes, dtype=str),
        predictions=prediction_array,
        targets=target_array,
        cell_types=np.asarray(cell_types, dtype=str),
    )
    os.replace(temporary, progress_path(output_dir))


def load_progress(
    output_dir: Path, expected_cell_types: Sequence[str]
) -> tuple[int, list[int], list[str], list[np.ndarray], list[np.ndarray]]:
    path = progress_path(output_dir)
    if not path.exists():
        return 0, [], [], [], []
    with np.load(path, allow_pickle=False) as state:
        saved_cell_types = state["cell_types"].tolist()
        if saved_cell_types != list(expected_cell_types):
            raise ValueError(
                "progress cell types do not match the current inputs; use a "
                "different --output-dir or pass --restart"
            )
        return (
            int(state["next_index"]),
            state["source_indices"].astype(int).tolist(),
            state["genes"].astype(str).tolist(),
            [row.copy() for row in state["predictions"]],
            [row.copy() for row in state["targets"]],
        )


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def write_final_outputs(
    output_dir: Path,
    source_indices: Sequence[int],
    genes: Sequence[str],
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    cell_types: Sequence[str],
    metadata: dict,
) -> None:
    prediction_array = np.stack(predictions).astype(np.float32, copy=False)
    target_array = np.stack(targets).astype(np.float32, copy=False)

    prediction_frame = pd.DataFrame(prediction_array, columns=cell_types)
    prediction_frame.insert(0, "gene", genes)
    prediction_frame.insert(0, "source_index", source_indices)
    target_frame = pd.DataFrame(target_array, columns=cell_types)
    target_frame.insert(0, "gene", genes)
    target_frame.insert(0, "source_index", source_indices)
    atomic_parquet(prediction_frame, output_dir / "predictions.parquet")
    atomic_parquet(target_frame, output_dir / "targets.parquet")

    log_predictions = np.log2(prediction_array + 1)
    log_targets = np.log2(target_array + 1)
    enough_rows = len(genes) >= 2
    correlations = pd.DataFrame(
        {
            "celltype": cell_types,
            "pearson": [
                scipy.stats.pearsonr(
                    log_predictions[:, i], log_targets[:, i]
                ).statistic
                if enough_rows
                else np.nan
                for i in range(len(cell_types))
            ],
            "spearman": [
                scipy.stats.spearmanr(
                    log_predictions[:, i], log_targets[:, i]
                ).statistic
                if enough_rows
                else np.nan
                for i in range(len(cell_types))
            ],
        }
    )
    temporary_csv = output_dir / ".per_cell_type_correlations.csv.tmp"
    correlations.to_csv(temporary_csv, index=False)
    os.replace(temporary_csv, output_dir / "per_cell_type_correlations.csv")

    metadata = {
        **metadata,
        "complete": True,
        "full_dataset_complete": metadata["requested_end_index"]
        == metadata["source_rows"],
        "evaluated_genes": len(genes),
        "prediction_shape": list(prediction_array.shape),
        "target_shape": list(target_array.shape),
    }
    temporary_json = output_dir / ".metadata.json.tmp"
    with open(temporary_json, "w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    os.replace(temporary_json, output_dir / "metadata.json")


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    submission_data = args.submission_data.expanduser().resolve()
    data_root = submission_data / "scooby_training_data"
    output_dir = args.output_dir.expanduser().resolve()
    paths = {
        "checkpoint weights": checkpoint / "model.safetensors",
        "checkpoint config": checkpoint / "scag_config.yaml",
        "fasta": data_root / "genome_human.fa",
        "test genes": data_root / "test_gene_sequences.csv",
        "gtf": submission_data
        / "reproducibility_data"
        / "gencode.v32.annotation.sorted.gtf.gz",
        "embedding": data_root / "embedding_no_val_genes_new.pq",
        "cell types": data_root / "celltype_fixed.pq",
        "pseudobulks": data_root / "pseudobulks",
    }
    require_paths(paths)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("NeurIPS scAG inference requires a CUDA GPU")
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    config = inference_config(checkpoint)
    embedding = pd.read_parquet(paths["embedding"])
    cell_type_index = normalize_cell_types(pd.read_parquet(paths["cell types"]))
    cell_types = cell_type_index["celltype"].astype(str).tolist()
    if len(embedding.iloc[0]["embedding"]) != config.model.cell_emb_dim:
        raise ValueError("embedding width does not match the checkpoint")
    for cell_type in cell_types:
        for strand in ("plus", "minus"):
            path = paths["pseudobulks"] / f"{strand}.{cell_type}.bw"
            if not path.exists():
                raise FileNotFoundError(path)

    if args.restart:
        next_index, source_indices, genes, predictions, targets = (
            0,
            [],
            [],
            [],
            [],
        )
    else:
        next_index, source_indices, genes, predictions, targets = load_progress(
            output_dir, cell_types
        )
    if next_index:
        print(
            f"Resuming at source row {next_index:,} with "
            f"{len(genes):,} completed genes"
        )

    print(f"Loading {config.run_name} from {checkpoint}")
    model = ScoobyAlphaGenome.from_pretrained(
        str(checkpoint), config=config, device=device, train=False
    )
    inner = unwrap(model)
    model.eval()
    # Cache the sequence embeddings so AlphaGenome runs twice per gene, not per cell type.
    inner.disable_cache = False
    inner.sequences, inner.last_embs = [], []
    cell_kernels = build_cell_kernels(inner, embedding, cell_type_index, device)

    print(f"Reading transcriptome annotation from {paths['gtf']}")
    transcriptome = Transcriptome(str(paths["gtf"]))
    test_ds = GenomeIntervalDataset(
        bed_file=str(paths["test genes"]),
        fasta_file=str(paths["fasta"]),
        filter_df_fn=lambda frame: frame.filter(pl.col("column_2") >= 0),
        return_seq_indices=False,
        shift_augs=(0, 0),
        rc_aug=False,
        return_augs=True,
        context_length=524_288,
        chr_bed_to_fasta_map={},
    )
    target_dataset = onTheFlyPseudobulkDataset(
        cell_types=cell_types,
        ds=test_ds,
        base_path=str(paths["pseudobulks"]),
    )
    end_index = len(test_ds)
    if args.max_genes is not None:
        end_index = min(end_index, args.max_genes)
    if next_index > end_index:
        raise ValueError(
            f"progress starts at {next_index}, beyond requested end {end_index}"
        )

    metadata = {
        "checkpoint": str(checkpoint),
        "submission_data": str(submission_data),
        "run_name": config.run_name,
        "source_rows": len(test_ds),
        "requested_end_index": end_index,
        "cell_types": cell_types,
    }

    current_index = next_index
    try:
        for i in tqdm(range(next_index, end_index), desc="Genes", initial=next_index, total=end_index):
            current_index = i
            gene = str(test_ds.df[i, "column_4"])
            position = test_ds.df[i, "column_2"]
            gene_slice, strand = get_gene_slice_and_strand(
                transcriptome, gene, position, span=False
            )
            if len(gene_slice) == 0:
                current_index = i + 1
                continue

            sequence, _rc_aug, target_profiles = target_dataset[i]
            sequence = sequence.unsqueeze(0).to(device)  # NLC for AlphaGenome
            predicted_counts = get_pseudobulk_count_pred(
                csb=model,
                seqs=sequence,
                cell_emb_conv_weights_and_biases=cell_kernels,
                gene_slice=gene_slice,
                strand=strand,
                model_type="multiome",
                predict=predict_with_rc,
                clip_soft=5.0,
                num_neighbors=1,
                chunk_size=args.cell_chunk_size,
            )

            strand_offset = 0 if strand == "+" else 1
            target_counts = undo_squashed_scale(
                target_profiles[gene_slice, strand_offset::2], clip_soft=384
            ).sum(axis=0)
            source_indices.append(i)
            genes.append(gene)
            predictions.append(predicted_counts.float().cpu().numpy())
            targets.append(target_counts.float().cpu().numpy())
            current_index = i + 1
            del sequence, target_profiles, predicted_counts, target_counts

            if current_index % args.save_every == 0:
                save_progress(
                    output_dir,
                    current_index,
                    source_indices,
                    genes,
                    predictions,
                    targets,
                    cell_types,
                )
    except BaseException:
        save_progress(
            output_dir,
            current_index,
            source_indices,
            genes,
            predictions,
            targets,
            cell_types,
        )
        print(f"Saved interrupted progress at source row {current_index:,}")
        raise

    save_progress(
        output_dir,
        end_index,
        source_indices,
        genes,
        predictions,
        targets,
        cell_types,
    )
    if not predictions:
        raise RuntimeError("no genes produced predictions")
    write_final_outputs(
        output_dir,
        source_indices,
        genes,
        predictions,
        targets,
        cell_types,
        metadata,
    )
    print(
        f"Saved {len(genes):,} genes x {len(cell_types)} cell types to {output_dir}"
    )


if __name__ == "__main__":
    main()
