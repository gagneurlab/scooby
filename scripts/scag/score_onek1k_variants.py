#!/usr/bin/env python3
"""Score fine-mapped OneK1K SNVs with scAG or Borzoi-based Scooby.

Uses the published high-PIP association table used in Figure4.ipynb. Each
variant-gene pair is scored once, while all tissue-specific associations are
retained for evaluation. Predictions are made per cell type of the supplied
cell-type index; matching them to OneK1K tissues is left to the plotting step.
Both multiome (3-track) and RNA-only (2-track) checkpoints are supported.
Progress resumes automatically when the config, pair list and cell types match.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

# scripts/scag/, which holds configs/ and the default results/ directory.
SCAG_ROOT = Path(__file__).resolve().parent

import numpy as np
import pandas as pd
from pyfaidx import Fasta
import yaml

CONTEXT = 524288
OUTPUT_LENGTH = 6144 * 32
FLANK = (CONTEXT - OUTPUT_LENGTH) // 2


def load_config(path):
    """Optionally inherit the input config, keeping both models' inputs identical."""
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if "base_config" in config:
        base_path = path.parent / config.pop("base_config")
        if base_path.resolve() == path.resolve():
            raise ValueError("Config cannot inherit itself")
        with base_path.open() as handle:
            base = yaml.safe_load(handle)
        if "base_config" in base:
            raise ValueError("Nested config inheritance is not supported")
        config = {**base, **config}
    if config.get("backend", "scag") not in ("scag", "scooby"):
        raise ValueError("backend must be scag or scooby")
    return config


def predict_borzoi_with_rc(model, seqs, seqs_rev_comp, conv_weights, conv_biases,
                           bins_to_predict=None):
    """Adapt shared NLC inputs to Borzoi's NCL layout, then average FWD/RC."""
    import torch
    from scooby.scag.predict import fix_rev_comp
    with torch.no_grad(), torch.autocast("cuda", enabled=seqs.is_cuda):
        outputs = model.forward_sequence_w_convs(
            seqs.permute(0, 2, 1), conv_weights, conv_biases,
            bins_to_predict=bins_to_predict)
        rc_bins = None if bins_to_predict is None else 6143 - bins_to_predict
        outputs_rc = model.forward_sequence_w_convs(
            seqs_rev_comp.permute(0, 2, 1), conv_weights, conv_biases,
            bins_to_predict=rc_bins)
    return (outputs + fix_rev_comp(torch.flip(outputs_rc, (1, -3)), model.n_tracks)) / 2


def select_associations(frame, study_label, pip_threshold):
    required = {"variant", "gene_id", "gene_name", "study_label", "pip",
                "Chromosome", "Pos", "Ref", "Alt"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Variant table is missing columns: {sorted(missing)}")
    selected = frame.loc[(frame.study_label == study_label)
                         & (frame.pip > pip_threshold)].copy()
    if selected.empty:
        raise ValueError("No associations pass study/PIP selection")
    if selected[list(required)].isna().any().any():
        raise ValueError("Missing required association metadata")
    selected["gene_id"] = selected.gene_id.str.split(".").str[0]
    selected["pair_id"] = selected.variant + "|" + selected.gene_id
    # The high-PIP table's Pos is zero-based; variant strings are one-based.
    for row in selected.itertuples(index=False):
        chrom, pos, ref, alt = row.variant.rsplit("_", 3)
        if (chrom != row.Chromosome or int(pos) != row.Pos + 1
                or ref != row.Ref or alt != row.Alt):
            raise ValueError(f"Inconsistent variant/zero-based Pos: {row.variant}")
    pairs = selected.drop_duplicates("pair_id").copy()
    identity = ["gene_name", "Chromosome", "Pos", "Ref", "Alt"]
    if (selected.groupby("pair_id")[identity].nunique() > 1).any().any():
        raise ValueError("Conflicting coordinates or gene names for a pair")
    return selected, pairs.reset_index(drop=True)


def read_window(fasta, chrom, start, end):
    """BED/pyfaidx zero-based, half-open; match enformer's '.' padding."""
    chromosome = fasta[chrom]
    size = len(chromosome)
    sequence = ("." * max(0, -start)
                + str(chromosome[max(0, start):min(size, end)]).upper()
                + "." * max(0, end - size))
    if len(sequence) != end - start:
        raise ValueError(f"Invalid FASTA window: {chrom}:{start}-{end}")
    return sequence


def mutate_snv(sequence, position, ref, alt):
    if ref not in "ACGT" or alt not in "ACGT" or len(ref) != 1 or len(alt) != 1:
        raise ValueError("Only unambiguous SNVs are supported")
    if not 0 <= position < len(sequence):
        raise ValueError("Variant is outside the sequence window")
    if sequence[position] != ref:
        raise ValueError(f"FASTA REF mismatch at offset {position}: "
                         f"expected {ref}, found {sequence[position]}")
    return sequence[:position] + alt + sequence[position + 1:]


def prepare_pairs(pairs, config, transcriptome, fasta):
    windows = pd.read_table(config["gene_windows"], header=None,
                            names=["window_chrom", "output_start", "output_end",
                                   "gene_name", "window_strand"])
    if windows.gene_name.duplicated().any():
        raise ValueError("Gene-window BED has ambiguous duplicate gene names")
    pairs = pairs.merge(windows, on="gene_name", how="left", validate="many_to_one")
    accepted, dropped, slices = [], [], {}
    for row in pairs.to_dict("records"):
        reason = None
        gene = transcriptome.genes.get(row["gene_name"])
        if len(row["Ref"]) != 1 or len(row["Alt"]) != 1 or row["Ref"] not in "ACGT" or row["Alt"] not in "ACGT":
            reason = "not_unambiguous_snv"
        elif pd.isna(row["output_start"]):
            reason = "no_gene_window"
        elif gene is None:
            reason = "no_gtf_gene"
        elif (row["Chromosome"] != row["window_chrom"]
              or gene.chrom != row["Chromosome"]
              or gene.strand != row["window_strand"]):
            reason = "gene_coordinate_mismatch"
        else:
            start, end = int(row["output_start"]), int(row["output_end"])
            if end - start != OUTPUT_LENGTH:
                raise ValueError(f"Unexpected output-window width for {row['gene_name']}")
            row.update(output_start=start, output_end=end,
                       sequence_start=start-FLANK, sequence_end=end+FLANK)
            if not row["sequence_start"] <= row["Pos"] < row["sequence_end"]:
                reason = "outside_input_window"
            else:
                bins = gene.output_slice(start, OUTPUT_LENGTH, 32, span=config["span"])
                if len(bins) == 0:
                    reason = "empty_gene_slice"
                else:
                    # Validate genomic REF independently of padding/window offsets.
                    pos = int(row["Pos"])
                    ref = str(fasta[row["Chromosome"]][pos:pos+1]).upper()
                    if ref != row["Ref"]:
                        raise ValueError(f"FASTA REF mismatch for {row['variant']}: {ref}")
                    slices[row["pair_id"]] = bins
        if reason:
            row["drop_reason"] = reason
            dropped.append(row)
        else:
            accepted.append(row)
    if not accepted:
        raise ValueError("No scoreable variant-gene pairs remain")
    return pd.DataFrame(accepted), pd.DataFrame(dropped), slices


def atomic_parquet(frame, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary)
    os.replace(temporary, path)


def save_state(path, signature, ref_counts, alt_counts):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, signature=signature,
                            ref_counts=ref_counts, alt_counts=alt_counts)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCAG_ROOT / "configs/onek1k_variants.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs/REFs without loading the model or writing outputs")
    parser.add_argument("--max-variants", type=int, help="Limit eligible variant-gene pairs for a GPU smoke test")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.max_variants is not None and args.max_variants < 1:
        parser.error("--max-variants must be positive")
    config = load_config(args.config)
    backend = config.get("backend", "scag")
    output = args.output_dir or Path(config["output_dir"])
    if not output.is_absolute():
        output = SCAG_ROOT / output
    config["output_dir"] = str(output)
    if config["save_every"] < 1:
        raise ValueError("save_every must be positive")
    checkpoint = Path(config["checkpoint"])
    for key in ("variants", "gene_windows", "gtf", "fasta", "embedding", "cell_type_index"):
        if not Path(config[key]).is_file():
            raise FileNotFoundError(f"Missing {key}: {config[key]}")
    if backend == "scag":
        for name in ("scag_config.yaml", "model.safetensors"):
            if not (checkpoint / name).is_file():
                raise FileNotFoundError(checkpoint / name)

    from scooby.utils.transcriptome import Transcriptome
    from tqdm.auto import tqdm
    associations, pairs = select_associations(pd.read_table(config["variants"]),
                                              config["study_label"], config["pip_threshold"])
    print(f"Selected {len(associations)} associations, {len(pairs)} unique variant-gene pairs", flush=True)
    transcriptome = Transcriptome(str(config["gtf"]))
    # Do not create or rebuild an index alongside the shared reference.
    fasta = Fasta(config["fasta"], build_index=False, rebuild=False)
    pairs, dropped, slices = prepare_pairs(pairs, config, transcriptome, fasta)
    print(f"Eligible pairs: {len(pairs)}; dropped: {len(dropped)}", flush=True)
    if not dropped.empty:
        print(dropped.drop_reason.value_counts().to_string(), flush=True)
    if args.max_variants:
        pairs = pairs.iloc[:args.max_variants].copy()
    # Validate the actual expanded windows and allele-injection offsets too.
    for row in tqdm(pairs.itertuples(index=False), total=len(pairs), desc="Validate windows"):
        sequence = read_window(fasta, row.Chromosome, row.sequence_start, row.sequence_end)
        mutate_snv(sequence, int(row.Pos)-row.sequence_start, row.Ref, row.Alt)
    if args.dry_run:
        fasta.close()
        print(f"Dry run OK: {len(pairs)} pairs validated; no model loaded or outputs written.")
        return

    import torch
    import anndata as ad
    from enformer_pytorch.data import str_to_one_hot
    from evaluate_neurips import inference_config, normalize_cell_types, build_cell_kernels
    from scooby.scag import ScoobyAlphaGenome, unwrap, predict_with_rc
    from scooby.utils.utils import get_pseudobulk_count_pred

    if not torch.cuda.is_available():
        raise RuntimeError("Scoring requires a CUDA GPU; use --dry-run for CPU validation")
    cell_index = normalize_cell_types(pd.read_parquet(config["cell_type_index"]))
    cell_types = cell_index.celltype.tolist()
    if len(set(cell_types)) != len(cell_types):
        raise ValueError("Duplicate cell-type labels")
    embedding = pd.read_parquet(config["embedding"])
    if backend == "scag":
        model_config = inference_config(checkpoint)
        if model_config.model.target_bins != 6144:
            raise ValueError("This scorer requires a 6144-bin checkpoint")
        n_tracks = model_config.model.num_tracks
        cell_emb_dim = model_config.model.cell_emb_dim
        checkpoint_config = (checkpoint / "scag_config.yaml").read_text()
    else:
        from scooby.modeling import Scooby
        model_config = Scooby.config_class.from_pretrained(
            str(checkpoint), revision=config.get("revision", "main"),
            cache_dir=config.get("cache_dir"), return_center_bins_only=True)
        if model_config.bins_to_return != 6144:
            raise ValueError("Borzoi checkpoint must predict 6144 center bins")
        # Defaults describe the published NeurIPS multiome model.
        n_tracks = config.get("n_tracks", 3)
        cell_emb_dim = config.get("cell_emb_dim", 14)
        use_transform_borzoi_emb = config.get("use_transform_borzoi_emb", True)
        checkpoint_config = model_config.to_json_string()
        # Pin weights to the commit whose config was just resolved.
        revision = getattr(model_config, "_commit_hash", None) or config.get("revision", "main")
    if n_tracks not in (2, 3):
        raise ValueError(f"Unsupported track count {n_tracks}; expected 2 (RNA) or 3 (multiome)")
    # Kernels are split along the (cell, track) axis, so a chunk must hold
    # whole cells.
    if config["cell_chunk_size"] < n_tracks or config["cell_chunk_size"] % n_tracks:
        raise ValueError(f"cell_chunk_size must be a positive multiple of {n_tracks}")
    model_type = "rna" if n_tracks == 2 else "multiome"
    if np.vstack(embedding.embedding.values).shape[1] != cell_emb_dim:
        raise ValueError("Cell embedding width differs from checkpoint")
    input_paths = [Path(config[key]) for key in
                   ("variants", "gene_windows", "gtf", "fasta", "embedding", "cell_type_index")]
    if (checkpoint / "model.safetensors").is_file():
        input_paths.append(checkpoint / "model.safetensors")
    metadata = {"config": config, "pair_ids": pairs.pair_id.tolist(),
                "cell_types": cell_types, "max_variants": args.max_variants,
                "pair_metadata_sha256": hashlib.sha256(pairs.to_json().encode()).hexdigest(),
                "input_files": {str(path): {"size": path.stat().st_size,
                                           "mtime_ns": path.stat().st_mtime_ns}
                                for path in input_paths},
                "effect": "log2(ALT RNA pseudobulk count + 1) - log2(REF RNA pseudobulk count + 1)",
                "rc_average": True, "position_convention": "zero-based Pos; one-based variant string",
                "checkpoint_config": checkpoint_config}
    if backend == "scooby":
        metadata["model_revision"] = revision
    signature = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "progress.npz"
    shape = (len(pairs), len(cell_types))
    if state_path.exists():
        with np.load(state_path, allow_pickle=False) as state:
            if str(state["signature"]) != signature:
                raise ValueError("Existing progress belongs to different inputs/config; choose a new --output-dir")
            ref_counts, alt_counts = state["ref_counts"].copy(), state["alt_counts"].copy()
        if ref_counts.shape != shape or alt_counts.shape != shape:
            raise ValueError("Invalid progress shape")
    else:
        if any(output.iterdir()):
            raise FileExistsError("Nonempty output directory without resumable state; choose another --output-dir")
        ref_counts = np.full(shape, np.nan, dtype=np.float32)
        alt_counts = np.full(shape, np.nan, dtype=np.float32)
        save_state(state_path, signature, ref_counts, alt_counts)
    atomic_parquet(pairs, output / "pairs.parquet")
    atomic_parquet(associations.loc[associations.pair_id.isin(pairs.pair_id)], output / "associations.parquet")
    atomic_parquet(dropped, output / "dropped_pairs.parquet")
    metadata_path = output / "metadata.json"
    temporary = metadata_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(metadata, indent=2))
    os.replace(temporary, metadata_path)

    todo = np.flatnonzero(~(np.isfinite(ref_counts).all(axis=1) & np.isfinite(alt_counts).all(axis=1)))
    if len(todo):
        device = torch.device("cuda")
        if backend == "scag":
            model = ScoobyAlphaGenome.from_pretrained(str(checkpoint), config=model_config, device=device, train=False)
            inner = unwrap(model)
            predict = predict_with_rc
        else:
            model, loading_info = Scooby.from_pretrained(
                str(checkpoint), config=model_config, revision=revision,
                cache_dir=config.get("cache_dir"), cell_emb_dim=cell_emb_dim,
                embedding_dim=1920, n_tracks=n_tracks, disable_cache=False,
                use_transform_borzoi_emb=use_transform_borzoi_emb,
                output_loading_info=True)
            mismatches = {key: loading_info.get(key) for key in
                          ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                          if loading_info.get(key)}
            if mismatches:
                raise RuntimeError(f"Scooby weights did not load exactly: {mismatches}")
            model = model.to(device)
            inner = model
            predict = predict_borzoi_with_rc
        model.eval()
        inner.disable_cache = False
        inner.sequences, inner.last_embs = [], []
        kernels = build_cell_kernels(inner, embedding, cell_index, device)
        try:
            with torch.inference_mode():
                for completed, i in enumerate(tqdm(todo, desc="Variant-gene pairs"), 1):
                    row = pairs.iloc[i]
                    sequence = read_window(fasta, row.Chromosome, int(row.sequence_start), int(row.sequence_end))
                    alternate = mutate_snv(sequence, int(row.Pos)-int(row.sequence_start), row.Ref, row.Alt)
                    counts = []
                    for allele in (sequence, alternate):
                        inner.sequences, inner.last_embs = [], []
                        seq = str_to_one_hot(allele).unsqueeze(0).to(device)
                        prediction = get_pseudobulk_count_pred(
                            model, seq, kernels, slices[row.pair_id], row.window_strand,
                            predict, config["clip_soft"], model_type,
                            num_neighbors=1, chunk_size=config["cell_chunk_size"])
                        values = prediction.float().cpu().numpy()
                        if not np.isfinite(values).all() or (values < 0).any():
                            raise ValueError(f"Invalid predicted counts for {row.pair_id}")
                        counts.append(values)
                    ref_counts[i], alt_counts[i] = counts
                    if completed % config["save_every"] == 0:
                        save_state(state_path, signature, ref_counts, alt_counts)
        finally:
            save_state(state_path, signature, ref_counts, alt_counts)
            fasta.close()
    else:
        fasta.close()
    effects = np.log2(alt_counts + 1) - np.log2(ref_counts + 1)
    obs = pairs.set_index("pair_id")
    var = cell_index[["celltype", "size"]].set_index("celltype")
    result = ad.AnnData(effects, obs=obs, var=var)
    result.layers["ref_counts"], result.layers["alt_counts"] = ref_counts, alt_counts
    result.uns["effect_definition"] = metadata["effect"]
    result.uns["checkpoint"] = str(checkpoint)
    temporary = output / "effects.tmp.h5ad"
    result.write_h5ad(temporary, compression="gzip")
    os.replace(temporary, output / "effects.h5ad")
    atomic_parquet(pd.DataFrame(effects, index=obs.index, columns=cell_types), output / "effects.parquet")
    print(f"Saved effects for {len(pairs)} pairs × {len(cell_types)} cell types to {output}")


if __name__ == "__main__":
    main()
