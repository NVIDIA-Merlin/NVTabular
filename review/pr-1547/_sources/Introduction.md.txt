## [NVTabular](https://github.com/NVIDIA/NVTabular)

[![PyPI](https://img.shields.io/pypi/v/NVTabular?color=orange&label=version)](https://pypi.python.org/pypi/NVTabular/)
[![LICENSE](https://img.shields.io/github/license/NVIDIA-Merlin/NVTabular)](https://github.com/NVIDIA-Merlin/NVTabular/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/documentation-blue.svg)](https://nvidia-merlin.github.io/NVTabular/main/Introduction.html)

[NVTabular](https://github.com/NVIDIA-Merlin/NVTabular) is a feature engineering and preprocessing library for tabular data that is designed to easily manipulate terabyte scale datasets and train deep learning (DL) based recommender systems. It provides high-level abstraction to simplify code and accelerates computation on the GPU using the [RAPIDS Dask-cuDF](https://github.com/rapidsai/cudf/tree/main/python/dask_cudf) library.

NVTabular is a component of [NVIDIA Merlin](https://developer.nvidia.com/nvidia-merlin), an open source framework for building and deploying recommender systems and works with the other Merlin components including [Merlin Models](https://github.com/NVIDIA-Merlin/models), [HugeCTR](https://github.com/NVIDIA/HugeCTR) and [Merlin Systems](https://github.com/NVIDIA-Merlin/systems) to provide end-to-end acceleration of recommender systems on the GPU. Extending beyond model training, with NVIDIA’s  [Triton Inference Server](https://github.com/NVIDIA/tensorrt-inference-server), the feature engineering and preprocessing steps performed on the data during training can be automatically applied to incoming data during inference.

<!-- <img src='https://developer.nvidia.com/blog/wp-content/uploads/2020/07/recommender-system-training-pipeline-1.png'/> -->

### Benefits

When training DL recommender systems, data scientists and machine learning (ML) engineers have been faced with the following challenges:

* **Huge Datasets**: Commercial recommenders are trained on huge datasets that may be several terabytes in scale.
* **Complex Data Feature Engineering and Preprocessing Pipelines**: Datasets need to be preprocessed and transformed so that they can be used with DL models and frameworks. In addition, feature engineering creates an extensive set of new features from existing ones, requiring multiple iterations to arrive at an optimal solution.
* **Input Bottleneck**: Data loading, if not well optimized, can be the slowest part of the training process, leading to under-utilization of high-throughput computing devices such as GPUs.
* **Extensive Repeated Experimentation**: The entire data engineering, training, and evaluation process can be repetitious and time consuming, requiring significant computational resources.

NVTabular alleviates these challenges and helps data scientists and ML engineers:

* process datasets that exceed GPU and CPU memory without having to worry about scale.
* focus on what to do with the data and not how to do it by using abstraction at the operation level.
* prepare datasets quickly and easily for experimentation so that more models can be trained.
* deploy models into production by providing faster dataset transformation

Learn more in the NVTabular [core features documentation](https://nvidia-merlin.github.io/NVTabular/main/core_features.html).

### Performance

When running NVTabular on the Criteo 1TB Click Logs Dataset using a single V100 32GB GPU, feature engineering and preprocessing was able to be completed in 13 minutes. Furthermore, when running NVTabular on a DGX-1 cluster with eight V100 GPUs, feature engineering and preprocessing was able to be completed within three minutes. Combined with [HugeCTR](http://www.github.com/NVIDIA/HugeCTR/), the dataset can be processed and a full model can be trained in only six minutes.

The performance of the Criteo DRLM workflow also demonstrates the effectiveness of the NVTabular library. The original ETL script provided in Numpy took over five days to complete. Combined with CPU training, the total iteration time is over one week. By optimizing the ETL code in Spark and running on a DGX-1 equivalent cluster, the time to complete feature engineering and preprocessing was reduced to three hours. Meanwhile, training was completed in one hour.

### Installation

NVTabular requires Python version 3.7+. Additionally, GPU support requires:

* CUDA version 11.0+
* NVIDIA Pascal GPU or later (Compute Capability >=6.0)
* NVIDIA driver 450.80.02+
* Linux or WSL

#### Installing NVTabular Using Conda

NVTabular can be installed with Anaconda from the ```nvidia``` channel by running the following command:

```
conda install -c nvidia -c rapidsai -c numba -c conda-forge nvtabular python=3.7 cudatoolkit=11.2
```

#### Installing NVTabular Using Pip

NVTabular can be installed with `pip` by running the following command:

```
pip install nvtabular
```

> Installing NVTabular with Pip causes NVTabular to run on the CPU only and might require installing additional dependencies manually.
> When you run NVTabular in one of our Docker containers, the dependencies are already installed.

#### Installing NVTabular with Docker

NVTabular Docker containers are available in the [NVIDIA Merlin container
repository](https://catalog.ngc.nvidia.com/?filters=&orderBy=scoreDESC&query=merlin).
The following table summarizes the key information about the containers:

| Container Name             | Container Location | Functionality |
| -------------------------- | ------------------ | ------------- |
| merlin-hugectr             |https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-hugectr         | NVTabular, HugeCTR, and Triton Inference |
| merlin-tensorflow          |https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow      | NVTabular, Tensorflow and Triton Inference |
| merlin-pytorch             |https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-pytorch         | NVTabular, PyTorch, and Triton Inference |

To use these Docker containers, you'll first need to install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) to provide GPU support for Docker. You can use the NGC links referenced in the table above to obtain more information about how to launch and run these containers. To obtain more information about the software and model versions that NVTabular supports per container, see [Support Matrix](https://github.com/NVIDIA/NVTabular/blob/main/docs/source/resources/support_matrix.rst).

### Notebook Examples and Tutorials

We provide a [collection of examples, use cases, and tutorials](https://github.com/NVIDIA-Merlin/NVTabular/tree/main/examples) as Jupyter notebooks covering:

* Feature engineering and preprocessing with NVTabular
* Advanced workflows with NVTabular
* Scaling to multi-GPU and multi-node systems
* Integrating NVTabular with HugeCTR
* Deploying to inference with Triton

### Feedback and Support

If you'd like to contribute to the library directly, see the [Contributing.md](https://github.com/NVIDIA/NVTabular/blob/main/CONTRIBUTING.md). We're particularly interested in contributions or feature requests for our feature engineering and preprocessing operations. To further advance our Merlin Roadmap, we encourage you to share all the details regarding your recommender system pipeline in this [survey](https://developer.nvidia.com/merlin-devzone-survey).

If you're interested in learning more about how NVTabular works, see
[our NVTabular documentation](https://nvidia-merlin.github.io/NVTabular/main/Introduction.html). We also have [API documentation](https://nvidia-merlin.github.io/NVTabular/main/api/index.html) that outlines the specifics of the available calls within the library.
