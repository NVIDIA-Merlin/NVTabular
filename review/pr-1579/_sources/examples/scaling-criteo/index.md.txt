# Scaling to Large Datasets with Criteo

Criteo provides the largest publicly available dataset for recommender systems.
The dataset is 1 TB uncompressed click logs of 4 billion examples.
The example notebooks show how to scale NVTabular in the following ways:

* Using multiple GPUs and multiple nodes with NVTabular for ETL.
* Training recommender system model with NVTabular dataloader for PyTorch.

Refer to the following notebooks:

* [Download and Convert](01-Download-Convert.ipynb)
* [ETL with NVTabular](02-ETL-with-NVTabular.ipynb)
* Training a model: [HugeCTR](03-Training-with-HugeCTR.ipynb) | [TensorFlow](03-Training-with-TF.ipynb) | [FastAI](03-Training-with-FastAI.ipynb)
* Use Triton Inference Server to serve a model: [HugeCTR](04-Triton-Inference-with-HugeCTR.ipynb) | [TensorFlow](04-Triton-Inference-with-TF.ipynb)
