Installation
======

Hardware requirements
---------------------

-  NVIDIA GPU (tested on A40), Linux, Python >= 3.12 (required by alphagenome-pytorch)

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
