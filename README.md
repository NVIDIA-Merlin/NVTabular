## nvTabular
==========

Recommender systems require massive datasets to train, particularly for deep learning based solutions.  The transformation of these datasets after ETL in order to prepare them for model training is particularly challenging.  Often the time taken to do steps such as feature engineering, categorical encoding and normalization of continuous variables exceed the time it takes to train a model.

nvTabular is a feature engineering and preprocessing library for tabular data designed to quickly and easily manipulate terabyte scale datasets used to train deep learning based recommender systems.  It provides a high level abstraction to simplify code and accelerates computation on the GPU using the [RAPIDS cuDF](https://github.com/rapidsai/cudf) library.

nvTabular is designed to support Data Scientists and ML Engineers trying to train deep learning recommender systems or other tabular data problems by allowing them to:

* Prepare datasets quickly and easily in order to experiment more and train more models.
* Work with datasets that exceed GPU and CPU memory without having to worry about scale.
* Think about what you want to do with the data, not how you have to do it, using our abstraction at the operator level.

It is also meant to help ML/Ops Engineers deploying models into production by providing:

* Faster dataset transformation, allowing for production models to be trained more frequently and kept up date helping improve responsiveness and model performance.
* Integration with model serving frameworks like NVidia’s Triton Inference Server to make model deployment easy.
* Statistical monitoring of the dataset for distributional shift and outlier detection during production training or inference.

The library is designed to be interoperable with both PyTorch and Tensorflow using batch dataloaders that we’ve developed as extensions of native framework code.  nvTabular provides the option to shuffle data during preprocessing, allowing the dataloader to load large contiguous chunks from files rather than individual elements.  This allows us to do per epoch shuffles orders of magnitude faster than a full shuffle of the dataset.  We have benchmarked our dataloader at 100x the baseline item by item PyTorch dataloader and 3x the Tensorflow batch dataloader, with several optimizations yet to come in that stack.

Extending beyond model training, we plan to provide integration with model serving frameworks like NVidia’s Triton Inference Server, creating a clear path to production inference for these models and allowing the feature engineering and preprocessing steps performed on the data during training to be easily and automatically applied to incoming data during inference.

Our goal is faster iteration on massive tabular datasets, both for experimentation during training, and also for production model responsiveness.  

To be clear, this is an early alpha release, and we have a long way to go.  We have a working framework, but our [operator set](Operators) is extremely limited and every day we’re developing new optimizations that help improve the performance of the library.  If you’re interested in working with us to help develop this library we’re looking for early collaborators and contributors.  In the coming months we’ll be optimizing existing operators, adding a full set of common feature engineering and preprocessing operators, and extending our backend to support multi-node and multi-gpu systems.  Please reach out via email or see our guide on contributions.  We are particularly interested in contributions or feature requests for feature engineering or preprocessing operators that you have found helpful in your own workflows.


### Getting Started
----------------
nvTabular is available in the NVidia container repository at the following location:

[Docker quickstart]

Within the container is the codebase, along with all of our dependencies, particularly [RAPIDS cuDF](https://github.com/rapidsai/cudf) and a range of [examples](examples).  The easiest way to get started is to simply launch the container above and explore the examples within.  It is designed to work with [Cuda 10.2](https://developer.nvidia.com/cuda-downloads).  As we mature more cuda versions will be supported.

[Conda Install]

If you wish to install the library yourself you can do so using the commands above.  The requirements for the library include:

### Examples and Tutorials:
--------------

A workflow demonstrating the preprocessing and dataloading components of nvTabular can be found in the [NVidia Recommender System tutorial](https://developer.nvidia.com/deep-learning-examples#rec-sys) on training [Facebook's Deep Learning Recommender Model (DLRM)](https://github.com/facebookresearch/dlrm/) on the [Criteo 1TB dataset](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/).

[ DLRM Criteo Workflow ](https://developer.nvidia.com/deep-learning-examples#rec-sys)

We also have a simple tutorial that demonstrates similar functionality on a much smaller dataset, providing a pipeline for the [Rossman store sales dataset](https://www.kaggle.com/c/rossmann-store-sales) fed into a [fast.ai tabular data model](https://docs.fast.ai/tabular.html).  

[ Rossman Store Sales ](examples/gpu_benchmark-rossmann.ipynb)

### Contributing
------------
If you wish to contribute to the library directly please see Contributing.md.  We are in particular interested in contributions or feature requests for feature engineering or preprocessing operators that you have found helpful in your own workflows.

#### Core Contributors

nvTabular is supported and maintained directly by a small team of nvidians with a love of recommender systems.  They are: Ben Frederickson, Alec Gunny, Even Oldridge, Julio Perez, Onur Yilmaz, and Rick Zamora.

### Learn More
------------

If you’re interested in learning more about how nvTabular works under the hood we have provided a [more detailed description of the core functionality](HowItWorks.md).

We also have [API documentation](link to come) that outlines in detail the specifics of the calls available within the library.
