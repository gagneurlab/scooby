.. scooby master file, created by
   sphinx-quickstart on Fri Oct 11 15:41:10 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

scooby
====================

Modeling multi-modal genomic profiles from DNA sequence at single-cell resolution
------

Scooby is the first model to predict scRNA-seq coverage and scATAC-seq insertion profiles along the genome at single-cell resolution. For this, it leverages the pre-trained multi-omics profile predictor Borzoi as a foundation model, equips it with a cell-specific decoder, and fine-tune its sequence embeddings. Specifically, the decoder is conditioned on the cell position in a precomputed single-cell embedding.


.. toctree::
   :maxdepth: 3

   readme_link.rst
   installation.rst
   example_notebooks.rst
