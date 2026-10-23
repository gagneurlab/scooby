# Scooby-2 (scAG): Scooby with an AlphaGenome backbone

Scooby-2, developed under the name scAG, predicts single-cell RNA and ATAC
coverage profiles from DNA sequence. It keeps Scooby's cell-state decoder and
replaces the Borzoi sequence model with
[alphagenome-pytorch](https://github.com/genomicsxai/alphagenome-pytorch).
The implementation is the `scooby.scag` subpackage; `ScoobyAlphaGenome` is also
exported as `scooby.ScoobyAlphaGenome`.

## Install

Python 3.12 or newer is required, because alphagenome-pytorch requires it.

```bash
pip install git+https://github.com/gagneurlab/scooby.git
# or, from a clone of this repository:
pip install -e .
```

## Predict

```python
from scooby import ScoobyAlphaGenome
model = ScoobyAlphaGenome.from_pretrained("checkpoint/", device="cuda")
```

`checkpoint/` is a directory holding `model.safetensors` and
`scag_config.yaml`. `from_pretrained` reads the architecture and LoRA scope
from `scag_config.yaml`, so a checkpoint carries everything needed to rebuild
it; the training data and AlphaGenome weights named in that file are not
needed. If the argument is not an existing local directory, it is taken as a
Hugging Face Hub repo id and downloaded (`revision=`, `cache_dir=` and `token=`
are passed on to the Hub). Checkpoints from before this package are read
through their `training_config.yaml`.

The returned model is wrapped by peft. `model(sequence, cell_embeddings)`
works directly; for anything else, reach the underlying module with `unwrap`.
Sequences are one-hot DNA in NLC layout `(batch, length, 4)`; unlike the
Borzoi-based Scooby, do not permute them.

```python
from scooby.scag import predict_with_rc, unwrap

inner = unwrap(model)                       # peft proxies only forward()
weights, biases = inner.forward_cell_embs_only(cell_embeddings)
profiles = predict_with_rc(model, seqs, seqs_rc, weights, biases)
```

## Train

```bash
cp scooby/scag/configs/example.yaml my_run.yaml   # edit the paths
accelerate launch --config_file scripts/scag/accelerate_config.yaml \
    -m scooby.scag.train --config my_run.yaml
```

Every path lives in the config file. Relative paths resolve against the config's
own directory, and all of them are checked when the config loads, so a typo is
reported before the model is built rather than an hour into the run.

`scooby/scag/configs/` holds the configs of the published models
(`neurips.yaml`, `neurips_all_folds.yaml`, `onek1k_all_folds_headnorm.yaml`)
and `example.yaml`. They are installed with the package
(`importlib.resources.files("scooby.scag") / "configs"`). Replace their
`/path/to/` placeholders with your data and AlphaGenome weights.
`scripts/scag/accelerate_config.yaml` is the 4-GPU launch config these files
assume.

## Evaluation scripts

`scripts/scag/` holds the evaluation scripts, their configs (`configs/`) and,
by default, their outputs (`results/`). Relative paths in these configs resolve
against `scripts/scag/`. Like the training configs, they use `/path/to/`
placeholders.

### NeurIPS held-out genes

```bash
python scripts/scag/evaluate_neurips.py --checkpoint path/to/checkpoint \
    --submission-data path/to/submission_data
```

This predicts pseudobulk expression per cell type for the NeurIPS test genes
and writes predictions, targets and per-cell-type correlations to
`scripts/scag/results/neurips_inference/`. It needs one CUDA GPU and resumes
after interruption.

### OneK1K fine-mapped variant effects

From `scripts/scag/`:

```bash
python score_onek1k_variants.py --config configs/onek1k_variants.yaml --dry-run
python score_onek1k_variants.py --config configs/onek1k_variants.yaml
```

The config selects OneK1K associations at PIP > 0.9 from
`all_highpip_susie.tsv`, the table used in the reproducibility Figure4 notebook,
and loads `scag_neurips_all_folds/final` with its saved architecture/LoRA config.
`configs/onek1k_variants_scag_onek1k_headnorm.yaml` scores the OneK1K model in
OneK1K cell states instead.
Scoring requires one CUDA GPU; do not use a multi-process `accelerate launch`.
For a smoke test, append `--max-variants 2 --output-dir results/onek1k_smoke`.

Outputs are in `results/onek1k_scag_neurips_all_folds/`: `effects.h5ad` has one
row per unique variant–gene pair, columns for NeurIPS cell types, and REF/ALT
RNA pseudobulk count layers. Effects are `log2(ALT + 1) - log2(REF + 1)` using
RC-averaged, unsquashed gene-span counts. `effects.parquet` contains the same
effect matrix; `associations.parquet` retains each selected tissue-specific
association (including beta and PIP), joinable by `pair_id`.
`dropped_pairs.parquet` records excluded indels, unmapped genes, or variants
outside the 524 kb input window. REF mismatches abort rather than being ignored.
Rerunning the same command resumes `progress.npz`; use a separate output
directory when changing inputs or running a smoke test. These are predictions
in NeurIPS cell states, not donor-matched OneK1K predictions; cell-type matching
for evaluation remains an explicit downstream choice.

Plot observed beta against predicted effect with an explicit broad-cell-type
mapping (editable in `configs/onek1k_beta_scatter.yaml`):

```bash
python plot_onek1k_beta.py --config configs/onek1k_beta_scatter.yaml
```

This writes pooled and per-cell-type scatters as SVG/PNG/PDF in the results
directory's `plots/` folder, plus plotted/excluded association tables and
descriptive correlations. One point is a variant–gene association in a matched
cell type; memory-specific and other unmatched labels are excluded by default.

For a comparison that matches the fine-tuning dataset and cell-state inputs,
score OneK1K variants with `johahi/neurips-scooby`, then plot:

```bash
python score_onek1k_variants.py --config configs/onek1k_variants_scooby_neurips.yaml
python plot_onek1k_beta.py --config configs/onek1k_beta_scatter_scooby_neurips.yaml
```

The inference config inherits the scAG input config, so it uses the same
fine-mapped variants, 524 kb windows, NeurIPS cell embeddings and grouping,
gene-span RNA counts, and RC averaging. It saves resumable results separately
in `results/onek1k_scooby_neurips/`; the plotting config uses the identical six
cell-type mappings and makes a paired-panel comparison on the same points.
The published model is loaded with `use_transform_borzoi_emb=True`, as in the
NeurIPS reproducibility evaluation. The first run downloads the checkpoint to
`cache_dir`; `--dry-run` does not load/download the model. Use one CUDA GPU.

`configs/onek1k_variants_scooby_onek1k.yaml` and
`configs/onek1k_beta_scatter_scooby_onek1k.yaml` do the same for the
Borzoi-based Scooby trained on OneK1K, compared against the OneK1K Scooby-2
model.

## Design notes

**Early exit at 32 bp.** AlphaGenome decodes to 1 bp through seven upsampling
levels. Scooby only needs 32 bp bins, so with `early_exit_32bp` the decoder
stops after the second block, where the representation is already at 32 bp, and
a small output embedder replaces `embedder_1bp`. The high-resolution blocks are
deleted rather than skipped.

**Convolutional adapters are standardized.** A LoRA branch added in parallel
to AlphaGenome's weight-standardized convolutions is under no scale control and
can saturate the decoder. Scooby-2 folds the update into the kernel instead, so
it is standardized along with the pretrained weights:

```text
y = conv(x, standardize(W + (alpha / r) B A))
```

Transformer layers are adapted by peft as usual. `standardized_conv_lora:
false` restores the parallel form.

**Head input norm.** `model.head_input_norm: true` normalises the sequence
embedding over channels before the cell-state decoder, so the decoder's input
scale does not follow drift in the backbone. The OneK1K model needs it; without
it, OneK1K training collapsed around the end of warm-up.

**Collapse guard.** Training aborts when the median of each step's largest
prediction over the last `collapse_patience` steps falls below
`collapse_threshold`, which is the signature of a saturated output head.

**Compilation.** `training.compile: true` compiles the model with
TorchInductor (about 1.7x faster per step on an A40, after a short warm-up).
Checkpoints are saved uncompiled and work with either setting.

## Layout

| Path | Contents |
| --- | --- |
| `scooby/scag/config.py` | Typed config, path resolution, validation |
| `scooby/scag/model.py` | `ScoobyAlphaGenome`, `from_pretrained`, `save_pretrained` |
| `scooby/scag/lora.py` | Adapter targets, `StandardizedConvLoRA`, `apply_lora` |
| `scooby/scag/data.py` | Dataset construction from a config |
| `scooby/scag/train.py` | Training loop and collapse guard |
| `scooby/scag/predict.py` | Reverse-complement averaged inference |
| `scooby/scag/configs/` | Training configs of the published models |
| `scripts/scag/` | Evaluation scripts, their configs, accelerate launch config |
| `tests/scag/` | Tests |

## Tests

From the repository root, with pytest installed (`pip install -e ".[test]"`):

```bash
pytest                # fast tests
pytest -m slow        # also builds a full AlphaGenome backbone
```
