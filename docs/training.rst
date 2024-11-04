Model training
======

Model training
--------------
-  Download training data (for neurips) from the Zenodo
   `repo <https://zenodo.org/records/14018495>`__
-  Adapt the scripts/train_config.yaml and scripts/config_multiome.py
-  Run the training from the conda environment with ``accelerate launch --config_file train_config.yaml train_multiome.py``
-  Adapt and use the Evaluator from docs/notebooks to eval the model
