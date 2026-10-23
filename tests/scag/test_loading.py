"""Tests for the public import path, packaged configs and checkpoint lookup.

None of these build an AlphaGenome backbone.
"""

import subprocess
import sys
from importlib import resources

import pytest
import yaml

from scooby.scag import model as scag_model
from scooby.scag.config import ScAGConfig


def test_top_level_export_is_the_scag_class():
    import scooby
    import scooby.modeling
    from scooby import ScoobyAlphaGenome

    assert ScoobyAlphaGenome is scag_model.ScoobyAlphaGenome
    assert scooby.modeling.ScoobyAlphaGenome is ScoobyAlphaGenome
    assert "ScoobyAlphaGenome" in dir(scooby)
    with pytest.raises(AttributeError):
        scooby.NoSuchModel


def test_import_scooby_is_lazy():
    """``import scooby`` alone must not pull in torch or alphagenome-pytorch."""
    code = (
        "import sys, scooby; "
        "print('torch' in sys.modules, 'alphagenome_pytorch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["False", "False"]


def packaged_configs():
    return sorted(
        p for p in (resources.files("scooby.scag") / "configs").iterdir()
        if p.name.endswith(".yaml")
    )


def test_training_configs_are_packaged():
    names = {p.name for p in packaged_configs()}
    assert {"example.yaml", "neurips.yaml", "neurips_all_folds.yaml",
            "onek1k_all_folds_headnorm.yaml"} <= names


@pytest.mark.parametrize("path", packaged_configs(), ids=lambda p: p.name)
def test_packaged_configs_parse(path):
    # Their data paths only exist on the cluster the models were trained on.
    with resources.as_file(path) as local:
        cfg = ScAGConfig.from_yaml(local, check_paths=False)
    assert cfg.model.num_tracks == cfg.data.num_tracks


def write_checkpoint_config(tmp_path):
    """A stored checkpoint config whose data and AlphaGenome files are absent."""
    config = {
        "output_dir": "/nonexistent/out",
        "run_name": "test",
        "model": {
            "cell_emb_dim": 10, "num_tracks": 2, "head_input_norm": True,
            "pretrained_model": "/nonexistent/model_all_folds.safetensors",
        },
        "data": {
            "fasta_file": "/nonexistent/genome.fa",
            "bed_file": "/nonexistent/sequences.bed",
            "adata_rna_plus": "/nonexistent/plus.h5ad",
            "adata_rna_minus": "/nonexistent/minus.h5ad",
            "adata_atac": None,
            "embedding": "/nonexistent/embedding.pq",
        },
    }
    (tmp_path / scag_model.CONFIG_FILENAME).write_text(yaml.safe_dump(config))
    return tmp_path


def test_checkpoint_config_does_not_need_training_files(tmp_path):
    checkpoint = write_checkpoint_config(tmp_path)
    with pytest.raises(ValueError, match="no such file"):
        ScAGConfig.from_yaml(checkpoint / scag_model.CONFIG_FILENAME)

    cfg = scag_model._load_checkpoint_config(str(checkpoint))
    assert cfg.model.num_tracks == 2
    assert cfg.model.head_input_norm is True


def test_structural_errors_are_still_reported_without_path_checks(tmp_path):
    path = write_checkpoint_config(tmp_path) / scag_model.CONFIG_FILENAME
    config = yaml.safe_load(path.read_text())
    config["model"]["num_tracks"] = 3  # RNA-only data yields 2
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="num_tracks"):
        ScAGConfig.from_yaml(path, check_paths=False)


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """Replace snapshot_download and record its calls."""
    import huggingface_hub

    calls = []
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    def snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        if repo_id == "someone/missing":
            raise RuntimeError("404 Repository Not Found")
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    return calls, snapshot


def test_local_directory_is_used_as_is(fake_hub, tmp_path):
    calls, _ = fake_hub
    assert scag_model.resolve_checkpoint(tmp_path) == str(tmp_path)
    assert calls == []


@pytest.mark.parametrize("path", ["/nonexistent/checkpoint", "./checkpoint",
                                  "checkpoint/", "a/b/c"])
def test_missing_local_path_is_not_sent_to_the_hub(fake_hub, path):
    calls, _ = fake_hub
    with pytest.raises(FileNotFoundError, match="no such checkpoint directory"):
        scag_model.resolve_checkpoint(path)
    assert calls == []


def test_repo_id_is_downloaded(fake_hub):
    calls, snapshot = fake_hub
    resolved = scag_model.resolve_checkpoint(
        "johahi/neurips-scooby-2", revision="abc123", cache_dir="/tmp/cache"
    )
    assert resolved == str(snapshot)
    (repo_id, kwargs), = calls
    assert repo_id == "johahi/neurips-scooby-2"
    assert kwargs["revision"] == "abc123" and kwargs["cache_dir"] == "/tmp/cache"
    assert scag_model.WEIGHTS_FILENAME in kwargs["allow_patterns"]
    assert scag_model.CONFIG_FILENAME in kwargs["allow_patterns"]


def test_failed_download_names_both_options(fake_hub):
    with pytest.raises(FileNotFoundError, match="not a local directory.*404"):
        scag_model.resolve_checkpoint("someone/missing")


def test_from_pretrained_reports_a_download_without_weights(fake_hub):
    """A Hub snapshot without model.safetensors fails before building a model."""
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        scag_model.ScoobyAlphaGenome.from_pretrained("johahi/neurips-scooby-2")
