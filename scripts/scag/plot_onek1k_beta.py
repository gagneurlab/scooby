#!/usr/bin/env python3
"""Plot observed OneK1K beta versus predicted effect, with explicit matching.

One point is a variant-gene association in a matched OneK1K cell type. Different
tissues can share a variant-gene pair; pooled correlations are descriptive, not
an independent-sample significance test. No identity line is drawn because the
observed beta and predicted log2 count change need not have the same scale.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import yaml

# scripts/scag/, which holds configs/ and the default results/ directory.
SCAG_ROOT = Path(__file__).resolve().parent


def read_reproducibility_effects(directory, pair_ids):
    """Read the Scooby H5AD chunks loaded by reproducibility Figure4.ipynb."""
    import anndata as ad
    paths = sorted(directory.glob("*.h5ad"))
    if not paths:
        raise FileNotFoundError(f"No H5AD prediction chunks in {directory}")
    frames = []
    columns = None
    for path in paths:
        data = ad.read_h5ad(path)
        if not {"variant", "gene_id"}.issubset(data.obs.columns):
            raise ValueError(f"Missing variant/gene_id in {path}")
        if not data.var_names.is_unique:
            raise ValueError(f"Duplicate cell types in {path}")
        if columns is None:
            columns = list(data.var_names)
        elif set(data.var_names) != set(columns):
            raise ValueError(f"Cell types differ between prediction chunks: {path}")
        keys = (data.obs.variant.astype(str) + "|"
                + data.obs.gene_id.astype(str).str.split(".").str[0])
        keep = keys.isin(pair_ids).to_numpy()
        matrix = data.X[keep]
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        frame = pd.DataFrame(matrix, index=pd.Index(keys[keep], name="pair_id"),
                             columns=data.var_names).reindex(columns=columns)
        frames.append(frame)
    effects = pd.concat(frames)
    if not effects.index.is_unique:
        raise ValueError("Duplicate variant-gene predictions across H5AD chunks")
    print(f"Loaded {len(effects)} requested pairs from {len(paths)} reproducibility H5AD chunks")
    return effects


def matched_associations(associations, effects, mapping):
    """Keep the largest-sample association per pair/tissue, matching Figure4."""
    if effects.index.name != "pair_id" or not effects.index.is_unique:
        raise ValueError("Effects must have a unique pair_id index")
    required = {"pair_id", "tissue_label", "beta"}
    if required - set(associations.columns):
        raise ValueError(f"Missing association columns: {required - set(associations.columns)}")
    if "sample_size" in associations:
        associations = associations.sort_values("sample_size", ascending=False, kind="stable")
    associations = associations.drop_duplicates(["pair_id", "tissue_label"]).copy()
    missing_pairs = set(associations.pair_id) - set(effects.index)
    if missing_pairs:
        raise ValueError(f"{len(missing_pairs)} association pair IDs are missing from effects")
    frames = []
    for tissue, columns in mapping.items():
        if not columns or len(set(columns)) != len(columns):
            raise ValueError(f"Empty or duplicate predicted cell types for {tissue}")
        missing = set(columns) - set(effects.columns)
        if missing:
            raise ValueError(f"Missing predicted cell types for {tissue}: {sorted(missing)}")
        subset = associations.loc[associations.tissue_label == tissue].copy()
        subset["pred_effect"] = effects.loc[subset.pair_id, columns].mean(axis=1).to_numpy()
        subset["predicted_cell_types"] = ";".join(columns)
        frames.append(subset)
    if not frames:
        raise ValueError("Empty cell-type mapping")
    matched = pd.concat(frames, ignore_index=True)
    excluded = associations.loc[~associations.tissue_label.isin(mapping)].copy()
    excluded["exclusion_reason"] = "no_configured_cell_type_match"
    finite = np.isfinite(matched.beta) & np.isfinite(matched.pred_effect)
    nonfinite = matched.loc[~finite].copy()
    nonfinite["exclusion_reason"] = "nonfinite_beta_or_prediction"
    excluded = pd.concat([excluded, nonfinite], ignore_index=True)
    matched = matched.loc[finite].copy()
    if matched.empty:
        raise ValueError("No finite, cell-type-matched associations")
    return matched, excluded


def correlation_summary(frame, label):
    varying = (len(frame) >= 3 and frame.beta.nunique() > 1
               and frame.pred_effect.nunique() > 1)
    return {"cell_type": label, "n_associations": len(frame),
            "n_unique_pairs": frame.pair_id.nunique(),
            "pearson_r": float(pearsonr(frame.beta, frame.pred_effect).statistic) if varying else np.nan,
            "spearman_rho": float(spearmanr(frame.beta, frame.pred_effect).statistic) if varying else np.nan}


def scatter(ax, frame, color):
    ax.axhline(0, color="0.7", lw=0.7, ls="--", zorder=0)
    ax.axvline(0, color="0.7", lw=0.7, ls="--", zorder=0)
    ax.scatter(frame.beta, frame.pred_effect, s=16, color=color,
               alpha=0.55, linewidths=0, zorder=2)
    summary = correlation_summary(frame, "")
    annotation = f"n = {len(frame):,}"
    if np.isfinite(summary["pearson_r"]):
        annotation += f"\nPearson r = {summary['pearson_r']:.2f}\nSpearman ρ = {summary['spearman_rho']:.2f}"
    ax.text(0.04, 0.96, annotation, transform=ax.transAxes, ha="left", va="top",
            fontsize=10, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=3))
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(fig, output, stem):
    for extension in ("svg", "png", "pdf"):
        fig.savefig(output / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCAG_ROOT / "configs/onek1k_beta_scatter.yaml")
    args = parser.parse_args()
    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else SCAG_ROOT / path
    results, output = resolve(config["results_dir"]), resolve(config["output_dir"])
    associations = pd.read_parquet(results / "associations.parquet")
    if "effects_h5ad_dir" in config:
        effects = read_reproducibility_effects(resolve(config["effects_h5ad_dir"]), set(associations.pair_id))
    else:
        effects = pd.read_parquet(results / "effects.parquet")
    matched, excluded = matched_associations(associations, effects, config["cell_type_mapping"])
    output.mkdir(parents=True, exist_ok=True)
    matched.to_csv(output / "beta_vs_pred_effect_points.tsv", sep="\t", index=False)
    excluded.to_csv(output / "beta_vs_pred_effect_excluded.tsv", sep="\t", index=False)
    summaries = [correlation_summary(matched, "Pooled")]
    summaries.extend(correlation_summary(matched.loc[matched.tissue_label == tissue], tissue)
                     for tissue in config["cell_type_mapping"])
    pd.DataFrame(summaries).to_csv(output / "beta_vs_pred_effect_summary.tsv", sep="\t", index=False)
    (output / "beta_vs_pred_effect_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12,
                         "axes.titlesize": 12, "svg.fonttype": "none",
                         "pdf.fonttype": 42})
    color = config.get("color", "crimson")
    model_label = config.get("model_label", "Scooby-2")
    ylabel = f"{model_label} predicted effect (Δ log₂ count)"
    fig, ax = plt.subplots(figsize=(5, 5))
    scatter(ax, matched, color)
    ax.set(xlabel="OneK1K eQTL beta", ylabel=ylabel,
           title="OneK1K · matched blood cell types")
    fig.tight_layout()
    save_figure(fig, output, "beta_vs_pred_effect")

    tissues = [tissue for tissue in config["cell_type_mapping"]
               if (matched.tissue_label == tissue).any()]
    ncols = min(3, len(tissues))
    nrows = int(np.ceil(len(tissues) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.7*nrows),
                             sharex=True, sharey=True, squeeze=False)
    for ax, tissue in zip(axes.flat, tissues):
        scatter(ax, matched.loc[matched.tissue_label == tissue], color)
        ax.set_title(tissue)
    for ax in list(axes.flat)[len(tissues):]:
        ax.set_visible(False)
    fig.supxlabel("OneK1K eQTL beta")
    fig.supylabel(ylabel)
    fig.tight_layout()
    save_figure(fig, output, "beta_vs_pred_effect_by_cell_type")

    if "reference_points" in config:
        reference = pd.read_table(resolve(config["reference_points"]))
        keys = ["pair_id", "tissue_label"]
        comparison = reference.merge(matched, on=keys, suffixes=("_reference", "_model"),
                                     validate="one_to_one")
        if len(comparison) != len(reference) or len(comparison) != len(matched):
            raise ValueError("Reference and model plots do not contain the same pair/cell-type set")
        if not np.allclose(comparison.beta_reference, comparison.beta_model):
            raise ValueError("Observed betas differ between reference and model")
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), sharex=True, sharey=True)
        reference_label = config.get("reference_label", "Scooby-2")
        for ax, frame, label, panel_color in (
            (axes[0], reference, reference_label, config.get("reference_color", "crimson")),
            (axes[1], matched, model_label, color),
        ):
            scatter(ax, frame, panel_color)
            ax.set_title(label)
        fig.supxlabel("OneK1K eQTL beta")
        fig.supylabel("Predicted effect (Δ log₂ count)")
        fig.tight_layout()
        save_figure(fig, output, "beta_vs_pred_effect_scag_vs_scooby")
        comparison.to_csv(output / "beta_vs_pred_effect_comparison.tsv", sep="\t", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"Plotted {len(matched)} matched associations; excluded {len(excluded)} unmatched/nonfinite associations.")
    print(f"Saved SVG, PNG, PDF and underlying data to {output}")


if __name__ == "__main__":
    main()
