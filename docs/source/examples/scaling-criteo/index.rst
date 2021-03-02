Scaling to Large Datasets with Criteo
=====================================

`Criteo <https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/>`_ provides the largest publicly available dataset for recommender systems, having a size of 1TB uncompressed click logs of 4 billion examples. We will teach to scale NVTabular:

* Using **multiple GPUs and multiple nodes** with NVTabular for ETL
* Training recommender system model with NVTabular dataloader for PyTorch

.. toctree::
   :maxdepth: 1

   Download and Convert <01-Download-Convert>
   ETL with NVTabular, Training with PyTorch <02-03b-ETL-with-NVTabular-Training-with-PyTorch>
   ETL with NVTabular, Training with HugeCTR <02-03c-ETL-with-NVTabular-HugeCTR.ipynb>
