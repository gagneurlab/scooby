"""Inference helpers."""

from __future__ import annotations

import torch

from scooby.utils.utils import fix_rev_comp_multiome, fix_rev_comp_rna

from .model import unwrap


def fix_rev_comp(outputs_rev_comp: torch.Tensor, n_tracks: int) -> torch.Tensor:
    """Swap the RNA strands of a reverse-complement pass back into place.

    Args:
        outputs_rev_comp: Flipped ``(B, bins, num_cells * n_tracks)`` output.
        n_tracks: 2 for RNA plus/minus, 3 with ATAC.
    """
    if n_tracks == 2:
        return fix_rev_comp_rna(outputs_rev_comp)
    if n_tracks == 3:
        return fix_rev_comp_multiome(outputs_rev_comp)
    raise ValueError(f"n_tracks must be 2 or 3, got {n_tracks}")


@torch.no_grad()
def predict_with_rc(
    model,
    seqs: torch.Tensor,
    seqs_rev_comp: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_biases: torch.Tensor,
    bins_to_predict=None,
    n_bins: int = 6144,
) -> torch.Tensor:
    """Average predictions over a sequence and its reverse complement.

    Args:
        model: A model from :meth:`ScoobyAlphaGenome.from_pretrained`, wrapped
            or not.
        seqs: One-hot DNA in NLC layout ``(B, S, 4)``. AlphaGenome takes NLC,
            so do not permute.
        seqs_rev_comp: The reverse complement of ``seqs``, same layout.
        conv_weights: Per-cell kernels from ``forward_cell_embs_only``.
        conv_biases: Per-cell biases from ``forward_cell_embs_only``.
        bins_to_predict: Optional bin subset. Indices are mirrored for the
            reverse-complement pass.
        n_bins: Total bins in the cropped output, used for that mirroring.

    Returns:
        The averaged prediction, ``(B, bins, num_cells * n_tracks)``.
    """
    inner = unwrap(model)
    with torch.autocast("cuda"):
        outputs = inner.forward_sequence_w_convs(
            seqs, conv_weights, conv_biases, bins_to_predict=bins_to_predict
        )
        rc_bins = None if bins_to_predict is None else (n_bins - 1 - bins_to_predict)
        outputs_rc = inner.forward_sequence_w_convs(
            seqs_rev_comp, conv_weights, conv_biases, bins_to_predict=rc_bins
        )
    flipped = fix_rev_comp(torch.flip(outputs_rc, (1, -3)), inner.n_tracks)
    return (outputs + flipped) / 2
