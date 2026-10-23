import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "scag"

spec = importlib.util.spec_from_file_location(
    "plot_onek1k_beta", SCRIPTS / "plot_onek1k_beta.py")
plotting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plotting)


def test_pair_alignment_average_and_unmatched_exclusion():
    associations = pd.DataFrame({"pair_id": ["p2", "p1", "p1"],
                                 "tissue_label": ["T", "T", "memory T"],
                                 "beta": [0.2, -0.3, 0.8]})
    effects = pd.DataFrame({"naive": [1., 2.], "activated": [3., 4.]},
                           index=pd.Index(["p1", "p2"], name="pair_id"))
    points, excluded = plotting.matched_associations(associations, effects, {"T": ["naive", "activated"]})
    assert points.pred_effect.tolist() == [3., 2.]
    assert excluded.tissue_label.tolist() == ["memory T"]


def test_duplicate_association_uses_largest_sample():
    associations = pd.DataFrame({"pair_id": ["p1", "p1"], "tissue_label": ["T", "T"],
                                 "beta": [0.2, 0.8], "sample_size": [10, 100]})
    effects = pd.DataFrame({"T": [1.]}, index=pd.Index(["p1"], name="pair_id"))
    points, _ = plotting.matched_associations(associations, effects, {"T": ["T"]})
    assert points.beta.tolist() == [0.8]


def test_missing_prediction_columns_fail():
    associations = pd.DataFrame({"pair_id": ["p1"], "tissue_label": ["T"], "beta": [0.2]})
    effects = pd.DataFrame({"T": [1.]}, index=pd.Index(["p1"], name="pair_id"))
    with pytest.raises(ValueError, match="Missing predicted cell types"):
        plotting.matched_associations(associations, effects, {"T": ["missing"]})


def test_constant_predictions_do_not_report_correlation():
    frame = pd.DataFrame({"pair_id": ["p1", "p2", "p3"], "beta": [1., 2., 3.],
                          "pred_effect": [0., 0., 0.]})
    assert np.isnan(plotting.correlation_summary(frame, "T")["pearson_r"])


def test_load_reproducibility_chunks_by_variant_gene_not_row_index(tmp_path):
    import anndata as ad
    for i, (variant, gene, value) in enumerate([("chr1_1_A_C", "ENSG1.2", 0.1),
                                               ("chr1_2_A_G", "ENSG2", 0.2)]):
        data = ad.AnnData(np.array([[value]]),
                         obs=pd.DataFrame({"variant": [variant], "gene_id": [gene]}, index=["0"]),
                         var=pd.DataFrame(index=["B_cell"]))
        data.write_h5ad(tmp_path / f"outs_{i}.h5ad")
    effects = plotting.read_reproducibility_effects(tmp_path, {"chr1_2_A_G|ENSG2"})
    assert effects.index.tolist() == ["chr1_2_A_G|ENSG2"]
    assert effects.iloc[0].B_cell == pytest.approx(0.2)
