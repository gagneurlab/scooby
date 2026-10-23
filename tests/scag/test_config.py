"""Tests for config loading, path resolution and validation."""

import os

import pytest
import yaml

from scooby.scag.config import ScAGConfig


def write_config(tmp_path, overrides=None, data_overrides=None):
    """Write a config whose declared files all exist inside ``tmp_path``."""
    for name in ("genome.fa", "sequences.bed", "plus.h5ad", "minus.h5ad",
                 "atac.h5ad", "embedding.pq"):
        (tmp_path / name).write_text("")

    config = {
        "output_dir": "out",
        "run_name": "test",
        "data": {
            "fasta_file": "genome.fa",
            "bed_file": "sequences.bed",
            "adata_rna_plus": "plus.h5ad",
            "adata_rna_minus": "minus.h5ad",
            "adata_atac": "atac.h5ad",
            "embedding": "embedding.pq",
        },
        "model": {"cell_emb_dim": 17},
    }
    config["data"].update(data_overrides or {})
    config.update(overrides or {})

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_relative_paths_resolve_against_the_config_file(tmp_path):
    cfg = ScAGConfig.from_yaml(write_config(tmp_path))
    assert cfg.data.fasta_file == str(tmp_path / "genome.fa")
    assert cfg.output_dir == str(tmp_path / "out")


def test_user_and_env_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("SCAG_TEST_ROOT", str(tmp_path))
    path = write_config(tmp_path, data_overrides={
        "fasta_file": "$SCAG_TEST_ROOT/genome.fa"
    })
    cfg = ScAGConfig.from_yaml(path)
    assert cfg.data.fasta_file == str(tmp_path / "genome.fa")


def test_missing_files_are_reported_together(tmp_path):
    path = write_config(tmp_path, data_overrides={
        "fasta_file": "absent.fa", "bed_file": "absent.bed"
    })
    with pytest.raises(ValueError) as excinfo:
        ScAGConfig.from_yaml(path)
    message = str(excinfo.value)
    assert "absent.fa" in message and "absent.bed" in message


def test_unknown_key_is_rejected(tmp_path):
    path = write_config(tmp_path, data_overrides={"fastaa_file": "genome.fa"})
    with pytest.raises(ValueError, match="fastaa_file"):
        ScAGConfig.from_yaml(path)


def test_empty_lora_scope_is_rejected(tmp_path):
    path = write_config(tmp_path, overrides={"model": {
        "cell_emb_dim": 17,
        "lora": {"encoder_convs": False, "decoder_convs": False,
                 "attn": False, "mlp": False},
    }})
    with pytest.raises(ValueError, match="LoRA"):
        ScAGConfig.from_yaml(path)


def test_context_length_must_match_encoder_stride(tmp_path):
    path = write_config(tmp_path, data_overrides={"context_length": 500_000})
    with pytest.raises(ValueError, match="multiple of 128"):
        ScAGConfig.from_yaml(path)


def test_round_trip_through_dump(tmp_path):
    cfg = ScAGConfig.from_yaml(write_config(tmp_path))
    out = tmp_path / "dumped.yaml"
    cfg.dump(out)
    reloaded = ScAGConfig.from_yaml(out)
    assert reloaded.to_dict() == cfg.to_dict()


def test_checkpoint_root(tmp_path):
    cfg = ScAGConfig.from_yaml(write_config(tmp_path))
    assert cfg.checkpoint_root == os.path.join(str(tmp_path / "out"), "test")


def test_unknown_compile_mode_is_rejected(tmp_path):
    path = write_config(tmp_path, overrides={
        "training": {"compile": True, "compile_mode": "turbo"},
    })
    with pytest.raises(ValueError, match="compile_mode"):
        ScAGConfig.from_yaml(path)


def test_compile_mode_is_ignored_when_compiling_is_off(tmp_path):
    path = write_config(tmp_path, overrides={
        "training": {"compile": False, "compile_mode": "turbo"},
    })
    assert ScAGConfig.from_yaml(path).training.compile is False


def test_compile_defaults_off(tmp_path):
    cfg = ScAGConfig.from_yaml(write_config(tmp_path))
    assert cfg.training.compile is False
    assert cfg.training.ddp_static_graph is True


def rna_only_config(tmp_path, num_tracks):
    path = write_config(tmp_path, overrides={"model": {
        "cell_emb_dim": 17, "num_tracks": num_tracks,
    }})
    config = yaml.safe_load(path.read_text())
    del config["data"]["adata_atac"]
    path.write_text(yaml.safe_dump(config))
    return path


def test_rna_only_config_without_atac(tmp_path):
    cfg = ScAGConfig.from_yaml(rna_only_config(tmp_path, num_tracks=2))
    assert cfg.data.rna_only
    assert cfg.data.num_tracks == 2
    assert set(cfg.data.adata_paths) == {"rna_plus", "rna_minus"}


def test_track_count_must_match_modalities(tmp_path):
    with pytest.raises(ValueError, match="num_tracks"):
        ScAGConfig.from_yaml(rna_only_config(tmp_path, num_tracks=3))
    with pytest.raises(ValueError, match="num_tracks"):
        ScAGConfig.from_yaml(write_config(tmp_path, overrides={"model": {
            "cell_emb_dim": 17, "num_tracks": 2,
        }}))
