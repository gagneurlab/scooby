Introduction
======

Code for scooby `manuscript <https://www.biorxiv.org/content/10.1101/2024.09.19.613754v2>`__. Scooby is the first model to predict
scRNA-seq coverage and scATAC-seq insertion profiles along the genome at
single-cell resolution. For this, it leverages the pre-trained
multi-omics profile predictor Borzoi as a foundation model, equips it
with a cell-specific decoder, and fine-tunes its sequence embeddings.
Specifically, the decoder is conditioned on the cell position in a
precomputed single-cell embedding.

This repository contains model and data loading code and a train script.
The reproducibility
`repository <https://github.com/gagneurlab/scooby_reproducibility>`__
contains notebooks to reproduce the results of the manuscript.

Hardware requirements
---------------------

-  NVIDIA GPU (tested on A40), Linux, Python (tested with v3.9)

Installation instructions
-------------------------

Prerequisites
~~~~~~~~~~~~~

scooby uses a a custom version of SnapATAC2, which can be installed with ``pip``. 

.. note::
   This is best installed in a separate environment due to numpy version conflicts with scooby.

-  ``pip install snapatac2-scooby``

Scooby package installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  ``pip install git+https://github.com/gagneurlab/scooby.git``
-  Download file contents from the Zenodo
   `repo <https://zenodo.org/records/13891693>`__
-  Use examples from the scooby reproducibility
   `repository <https://github.com/gagneurlab/scooby_reproducibility>`__

Training
--------

We offer a `train
script <https://github.com/gagneurlab/scooby/blob/main/scripts/train.py>`__,
which requires SNAPATAC2-preprocessed anndatas and embeddings. Training
takes 1-2 days on 8 NVIDIA A40 GPUs with 128GB RAM and 32 cores.

Model architecture
------------------

Currently, the model is only tested with a batch size of 1.

.. raw:: html

   <img width="500" alt="image" src="https://github.com/user-attachments/assets/98a52a86-67b1-4fb2-8b94-227ce2e47af2">
