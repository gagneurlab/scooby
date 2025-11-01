Scooby
======

.. raw:: html

   <img width="150" alt="image" src="https://scooby.readthedocs.io/en/latest/_static/logo.png">
.. image:: https://readthedocs.org/projects/scooby/badge/?version=latest
    :target: https://scooby.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

Code for the scooby `model <https://www.nature.com/articles/s41592-025-02854-5>`__. Scooby is the first model to predict
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

scooby uses a a custom version of SnapATAC2, which can be installed with ``pip``. This is best installed in a separate environment due to numpy version conflicts with scooby.

-  ``pip install snapatac2-scooby``

Scooby package installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  ``pip install git+https://github.com/gagneurlab/scooby.git``
-  Download file contents from Zenodo (`Training data <https://doi.org/10.5281/zenodo.15517072>`__ and `Resources <https://doi.org/10.5281/zenodo.15517764>`__)
-  Use examples from the scooby reproducibility
   `repository <https://github.com/gagneurlab/scooby_reproducibility>`__

Training
--------

We offer a `train
script for modeling scRNA-seq only <https://github.com/gagneurlab/scooby/blob/main/scripts/train_rna_only.py>`__ and a `script for multiome modeling <https://github.com/gagneurlab/scooby/blob/main/scripts/train_multiome.py>`__.
Both require SNAPATAC2-preprocessed anndatas and embeddings. Training scooby
takes 1-2 days on 8 NVIDIA A40 GPUs with 128GB RAM and 32 cores.

Pretrained Models
------------------

You can find pretrained models for the three datasets of the manuscript on Hugging Face:

-  `Neurips dataset <https://huggingface.co/johahi/neurips-scooby>`__
-  `OneK1K dataset <https://huggingface.co/lauradmartens/onek1k-scooby>`__
-  `Epicardioids dataset <https://huggingface.co/lauradmartens/epicardioids-scooby>`__

Model architecture
------------------

Currently, the model is only tested with a batch size of 1.

.. raw:: html

   <img width="500" alt="image" src="https://github.com/user-attachments/assets/98a52a86-67b1-4fb2-8b94-227ce2e47af2">
