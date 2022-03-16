Scaling to Large Datasets with Criteo
=====================================

`Criteo <https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/>`_ provides the largest publicly available dataset for recommender systems, having a size of 1TB uncompressed click logs of 4 billion examples. We will teach to scale NVTabular:

* Using **multiple GPUs and multiple nodes** with NVTabular for ETL
* Training recommender system model with NVTabular dataloader for PyTorch

.. toctree::
   :maxdepth: 1

   Download and Convert <01-Download-Convert>
   ETL with NVTabular <02-ETL-with-NVTabular>
   Training with TensorFlow <03-Training-with-TF>
   Training with HugeCTR <03-Training-with-HugeCTR>
   Training with FastAI <03-Training-with-FastAI>
   Triton Inference with TensorFlow <04-Triton-Inference-with-TF>
   Triton Inference with HugeCTR <04-Triton-Inference-with-HugeCTR>
