## [NVTabular](https://github.com/NVIDIA/NVTabular) | [API documentation](https://nvidia.github.io/NVTabular/main/Introduction.html)

[NVTabular](https://github.com/NVIDIA/NVTabular) is a feature engineering and preprocessing library for tabular data that is designed to quickly and easily manipulate terabyte scale datasets and train deep learning (DL) based recommender systems. It provides a high level abstraction to simplify code and accelerates computation on the GPU using the [RAPIDS Dask-cuDF](https://github.com/rapidsai/cudf/tree/main/python/dask_cudf) library. NVTabular is designed to be interoperable with both PyTorch and TensorFlow using dataloaders that we have developed as extensions of native framework code. In our experiments, we were able to speed up existing TensorFlow pipelines by 9 times and existing PyTorch pipelines by 5 times with our highly optimized dataloaders.

NVTabular is a component of [NVIDIA Merlin Open Beta](https://developer.nvidia.com/nvidia-merlin). NVIDIA Merlin is used for building large-scale recommender systems, which require massive datasets to train, particularly for deep learning based solutions. With NVTabular being a part of the Merlin ecosystem, it also works with the other Merlin components including [HugeCTR](https://github.com/NVIDIA/HugeCTR) and [Triton Inference Server](https://github.com/NVIDIA/tensorrt-inference-server) to provide end-to-end acceleration of recommender systems on the GPU. Extending beyond model training, with NVIDIA’s Triton Inference Server, the feature engineering and preprocessing steps performed on the data during training can be automatically applied to incoming data during inference.

### Benefits

Our ultimate goal is faster iteration on massive tabular datasets, both for experimentation during training, and also production model responsiveness. NVTabular is designed to support data scientists and machine learning (ML) engineers train (deep learning) recommender systems and resolve tabular data problems by allowing them to:

* prepare datasets quickly and easily for experimentation so that more models can be trained.
* process datasets that exceed GPU and CPU memory without having to worry about scale.
* use optimized dataloaders to accelerate training with TensorFlow, PyTorch, and HugeCTR.
* focus on what to do with the data and not how to do it by using abstraction at the operation level.

NVTabular also helps ML/Ops engineers with deploying models into production by providing faster dataset transformation. This makes it easy for production models to be trained more frequently and kept up to date, helping improve responsiveness and model performance.

To learn more about NVTabular's core features, see the following:
* [TensorFlow and PyTorch Interoperability](docs/source/core_features.md#tensorflow-and-pytorch-interoperability)
* [HugeCTR Interoperability](docs/source/core_features.md#hugectr-interoperability)
* [Multi-GPU Support](docs/source/core_features.md#multi-gpu-support)
* [Multi-Node Support](docs/source/core_features.md#multi-node-support)
* [Multi-Hot Encoding and Pre-existing Embeddings](docs/source/core_features.md#multi-hot-encoding-and-pre-existing-embeddings)
* [Shuffling Datasets](docs/source/core_features.md#shuffling-datasets)
* [Cloud Integration](docs/source/core_features.md#cloud-integration)

### Installation

To install NVTabular, ensure that you meet the following prerequisites:
* CUDA version 10.1+
* Python version 3.7+
* NVIDIA Pascal GPU or later

**NOTE**: NVTabular will only run on Linux. Other operating systems are not currently supported.

#### Installing NVTabular Using Conda

NVTabular can be installed with Anaconda from the ```nvidia``` channel:

```
conda install -c nvidia -c rapidsai -c numba -c conda-forge nvtabular python=3.7 cudatoolkit=10.2
```

If you'd like to create a full conda environment to run the example notebooks, you can use the [provided environment files](https://github.com/NVIDIA/NVTabular/tree/main/conda/environments) for CUDA Toolkit 10.1, 10.2, or 11.0. Clone the NVTabular repo and from the root directory, run:

```
conda env create -f=conda/environments/nvtabular_dev_cuda10.1.yml
conda activate nvtabular_dev_10.1
python -m ipykernel install --user --name=nvt
pip install -e .
jupyter notebook
```
Then open a notebook and select `nvt` from the `Kernel->Change Kernel` menu.

#### Installing NVTabular with Docker

Docker containers with NVTabular are available at the [NVIDIA Merlin container repository](https://ngc.nvidia.com/catalog/containers/nvidia:merlin). There are four different containers, each providing different functionality:


| Container Name             | Container Location | Functionality |
| -------------------------- | ------------------ | ------------- |
| merlin-training            | https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-training            | NVTabular and HugeCTR                                         |
| merlin-tensorflow-training | https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-tensorflow-training | NVTabular, TensorFlow and HugeCTR Tensorflow Embedding plugin |
| merlin-pytorch-training    | https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-pytorch-training    | NVTabular and PyTorch                                         |
| merlin-inference           | https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-inference           | NVTabular, HugeCTR and Triton Inference                       |

To use these Docker Containers, you'll first need to install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) to provide GPU support to docker.
There are more details on launching and running these containers on the NGC links above.

### Examples and Tutorials

We provide a [collection of examples, use cases, and tutorials](https://github.com/NVIDIA/NVTabular/tree/main/examples) as Jupyter notebooks in our repository. These Jupyter notebooks are based on the following datasets:

- MovieLens
- Outbrain Click Prediction
- Criteo Click Ads Prediction
- RecSys2020 Competition Hosted by Twitter
- Rossmann Sales Prediction

Each Jupyter notebook covers the following:

- Preprocessing and feature engineering with NVTabular
- Advanced workflows with NVTabular
- Accelerated dataloaders for TensorFlow and PyTorch
- Scaling to multi-GPU and multi nodes systems
- Integrating NVTabular with HugeCTR
- Deploying to inference with Triton

Performance of the Criteo DRLM workflow demonstrates the effectiveness of the NVTabular library. The original ETL script provided in Numpy took over five days to complete. Combined with CPU training, the total iteration time is over one week. By optimizing the ETL code in Spark and running on a DGX-1 equivalent cluster, we were able to bring that time down to three hours for ETL and one hour for training.

With NVTabular running on a single V100 32GB GPU, we were able to complete ETL in 13 minutes. With a DGX-1 cluster of eight V100 GPUs, we can accelerate ETL to 3 minutes. Combined with [HugeCTR](http://www.github.com/NVIDIA/HugeCTR/), we can process the dataset and train the full model in only 6 minutes. This fast iteration is the goal of NVTabular and the [Merlin application framework](https://developer.nvidia.com/nvidia-merlin). Additional information can be found [here](https://github.com/NVIDIA/NVTabular/tree/main/examples).

### Feedback and Support

If you'd like to contribute to the library directly, please see the [Contributing.md](https://github.com/NVIDIA/NVTabular/blob/main/CONTRIBUTING.md). We're particularly interested in contributions or feature requests for our feature engineering and preprocessing operations. To further advance our Merlin Roadmap, we encourage you to share all the details regarding your recommender system pipeline using this [this survey](https://developer.nvidia.com/merlin-devzone-survey).

If you're interested in learning more about how NVTabular works under the hood, see
[Architecture](https://nvidia.github.io/NVTabular/main/resources/api/index.html). We also have [API documentation](https://nvidia.github.io/NVTabular/main/resources/api/index.html) that outlines the specifics of the available calls within the library.
