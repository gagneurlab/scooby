"""Tests for reverse-complement track reordering."""

import pytest
import torch

from scooby.scag.predict import fix_rev_comp


def test_rna_only_swaps_strands_per_cell():
    # Two cells, tracks laid out as (cell, [plus, minus]).
    x = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]])
    assert fix_rev_comp(x, 2).tolist() == [[[1.0, 0.0, 3.0, 2.0]]]


def test_multiome_swaps_strands_and_keeps_atac():
    x = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]])
    assert fix_rev_comp(x, 3).tolist() == [[[1.0, 0.0, 2.0, 4.0, 3.0, 5.0]]]


def test_unknown_track_count_is_rejected():
    with pytest.raises(ValueError):
        fix_rev_comp(torch.zeros(1, 1, 4), 4)
