# Scooby

Code for scooby [manuscript]().
Scooby is the first model to predict scRNA-seq coverage and scATAC-seq insertion profiles along the genome at single-cell resolution.
For this, it leverages the pre-trained multi-omics profile predictor Borzoi as a foundation model, equips it with a cell-specific decoder, and fine-tune its sequence embeddings. Specifically, the decoder is conditioned on the cell position in a precomputed single-cell embedding.

This repository contains model and data loading code and a train script. 
The reproducibility [repository](https://github.com/gagneurlab/scooby_reproducibility) contains notebooks to reproduce the results of the manuscript.

## System and hardware requirements
 - NVIDIA GPU (tested on A40), Linux, Python (tested with v3.9)
 - PyTorch, tested with v2.1
 - scanpy, tested with v1.10
 - [borzoi-pytorch](https://github.com/johahi/borzoi-pytorch)
 - [SnapATAC2](https://github.com/lauradmartens/SnapATAC2)
 - [peft v0.10.1](https://github.com/lauradmartens/peft)

## Installation instructions
Installation will take roughly 30mins using conda.
 - Install above packages
 - Download file contents from Zenodo (link: TODO)
 - Use examples from tge scooby reproducibility [repository](https://github.com/gagneurlab/scooby_reproducibility)

## Training 
We offer a [train script](https://github.com/gagneurlab/scooby/blob/main/scripts/train.py), which requires SNAPATAC2-preprocessed adatas and embeddings.

## Model architecture
<img width="500" alt="image" src="https://github.com/user-attachments/assets/98a52a86-67b1-4fb2-8b94-227ce2e47af2">
