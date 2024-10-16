Installation
======

Hardware requirements
---------------------

-  NVIDIA GPU (tested on A40), Linux, Python (tested with v3.9)

Installation instructions
-------------------------

Prerequisites
~~~~~~~~~~~~~

scooby uses a a custom version of SnapATAC2, which we built using rust:

-  Install rust with
   ``curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh``
-  ``pip install git+https://github.com/lauradmartens/SnapATAC2.git#egg=snapatac2&subdirectory=snapatac2-python``

Scooby package installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  ``pip install git+https://github.com/gagneurlab/scooby.git``
-  Download file contents from the Zenodo
   `repo <https://zenodo.org/records/13891693>`__
-  Use examples from the scooby reproducibility
   `repository <https://github.com/gagneurlab/scooby_reproducibility>`__
