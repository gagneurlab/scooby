"""CPU checks for fine-mapped variant selection and genomic coordinates."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "scag"

spec = importlib.util.spec_from_file_location(
    "score_onek1k_variants", SCRIPTS / "score_onek1k_variants.py",
)
scoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scoring)


def association(**overrides):
    row = dict(variant="chr1_11_A_C", gene_id="ENSG1.2", gene_name="GENE",
               study_label="OneK1K", pip=0.95, Chromosome="chr1", Pos=10,
               Ref="A", Alt="C", tissue_label="B cell")
    row.update(overrides)
    return row


def test_keep_tissue_associations_score_unique_pairs():
    frame = pd.DataFrame([association(), association(tissue_label="T cell"),
                          association(study_label="Other"), association(pip=0.9)])
    associations, pairs = scoring.select_associations(frame, "OneK1K", 0.9)
    assert len(associations) == 2
    assert len(pairs) == 1
    assert pairs.iloc[0].pair_id == "chr1_11_A_C|ENSG1"


def test_detect_one_based_pos():
    with pytest.raises(ValueError, match="zero-based Pos"):
        scoring.select_associations(pd.DataFrame([association(Pos=11)]), "OneK1K", 0.9)


def test_mutate_only_requested_base_and_fail_on_wrong_ref():
    assert scoring.mutate_snv(".ACGT.", 1, "A", "T") == ".TCGT."
    with pytest.raises(ValueError, match="REF mismatch"):
        scoring.mutate_snv("ACGT", 1, "A", "T")
    with pytest.raises(ValueError, match="Only unambiguous SNVs"):
        scoring.mutate_snv("ACGT", 1, "CG", "T")


def test_fasta_window_padding_and_offsets(tmp_path):
    reference = tmp_path / "genome.fa"
    reference.write_text(">chr1\nACGT\n")
    with scoring.Fasta(str(reference)) as fasta:
        sequence = scoring.read_window(fasta, "chr1", -2, 6)
    assert sequence == "..ACGT.."
    assert scoring.mutate_snv(sequence, 0 - (-2), "A", "C") == "..CCGT.."


def test_neurips_baseline_inherits_identical_scag_inputs():
    original = scoring.load_config(SCRIPTS / "configs/onek1k_variants.yaml")
    baseline = scoring.load_config(SCRIPTS / "configs/onek1k_variants_scooby_neurips.yaml")
    assert baseline["checkpoint"] == "johahi/neurips-scooby"
    assert baseline["backend"] == "scooby"
    assert baseline["output_dir"] != original["output_dir"]
    for key in ("variants", "study_label", "pip_threshold", "gene_windows", "gtf",
                "fasta", "embedding", "cell_type_index", "span", "clip_soft"):
        assert baseline[key] == original[key]


def test_borzoi_adapter_layout_bins_and_rc_strand_swap():
    import numpy as np
    import torch

    class DummyBorzoi:
        n_tracks = 3

        def __init__(self):
            self.calls = []

        def forward_sequence_w_convs(self, sequence, weights, biases, bins_to_predict=None):
            self.calls.append((sequence.clone(), bins_to_predict.copy()))
            return torch.tensor([[[1., 2., 3.], [4., 5., 6.]]])

    model = DummyBorzoi()
    sequence = torch.arange(40).reshape(1, 10, 4).float()
    rc = sequence.flip((1, 2))
    bins = np.array([10, 20])
    result = scoring.predict_borzoi_with_rc(model, sequence, rc, None, None, bins)
    assert model.calls[0][0].shape == (1, 4, 10)
    assert torch.equal(model.calls[0][0], sequence.permute(0, 2, 1))
    assert torch.equal(model.calls[1][0], rc.permute(0, 2, 1))
    np.testing.assert_array_equal(model.calls[1][1], [6133, 6123])
    # Reverse bins and swap positive/negative RNA, but retain ATAC ordering.
    torch.testing.assert_close(result, torch.tensor([[[3., 3., 4.5], [3., 3., 4.5]]]))
