# Scooby

Code for scooby [manuscript]().
Scooby is the first model to predict scRNA-seq coverage and scATAC-seq insertion profiles along the genome at single-cell resolution.
For this, it leverages the pre-trained multi-omics profile predictor Borzoi as a foundation model, equips it with a cell-specific decoder, and fine-tune its sequence embeddings. Specifically, the decoder is conditioned on the cell position in a precomputed single-cell embedding.

This repository contains model and data loading code and a train script. 
The reproducibility [repository](https://github.com/gagneurlab/scooby_reproducibility) contains notebooks to reproduce the results of the manuscript.

## Model architecture
<img width="500" alt="image" src="https://github.com/user-attachments/assets/98a52a86-67b1-4fb2-8b94-227ce2e47af2">
