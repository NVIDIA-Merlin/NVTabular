## NVTabular

NVTabular is a feature engineering and preprocessing library for tabular data designed to quickly and easily manipulate terabyte scale datasets used to train deep learning based recommender systems. It provides a high level abstraction to simplify code and accelerates computation on the GPU using the [RAPIDS cuDF](https://github.com/rapidsai/cudf) library.

Recommender systems require massive datasets to train, particularly for deep learning based solutions. The transformation of these datasets after ETL in order to prepare them for model training is particularly challenging. Often the time taken to do steps such as feature engineering, categorical encoding and normalization of continuous variables exceeds the time it takes to train a model.

NVTabular is designed to support Data Scientists and ML Engineers trying to train (deep learning) recommender systems or other tabular data problems by allowing them to:

* Prepare datasets quickly and easily in order to experiment more and train more models.
* Work with datasets that exceed GPU and CPU memory without having to worry about scale.
* Focus on what to do with the data, and not how to do it, using our abstraction at the operation level.

It is also meant to help ML/Ops Engineers deploying models into production by providing:

* Faster dataset transformation, allowing for production models to be trained more frequently and kept up to date helping improve responsiveness and model performance.
* Integration with model serving frameworks like NVIDIA’s Triton Inference Server to make model deployment easy.
* Statistical monitoring of the dataset for distributional shift and outlier detection during production training or inference.

The library is designed to be interoperable with both PyTorch and Tensorflow using batch data-loaders that we have developed as extensions of native framework code. NVTabular provides the option to shuffle data during preprocessing, allowing the data-loader to load large contiguous chunks from files rather than individual elements. This allows us to do per epoch shuffles orders of magnitude faster than a full shuffle of the dataset. We have benchmarked our data-loader at 100x the baseline item by item PyTorch dataloader and 3x the Tensorflow batch data-loader, with several optimizations yet to come in that stack.

Extending beyond model training, we plan to provide integration with model serving frameworks like [NVIDIA’s Triton Inference Server](https://github.com/NVIDIA/tensorrt-inference-server), creating a clear path to production inference for these models and allowing the feature engineering and preprocessing steps performed on the data during training to be easily and automatically applied to incoming data during inference.

Our goal is faster iteration on massive tabular datasets, both for experimentation during training, and also for production model responsiveness. 

### Getting Started
NVTabular is available in the NVIDIA container repository at the following location, http://ngc.nvidia.com/catalog/containers/nvidia:nvtabular.

Currently we have the alpha release (0.1) container, you can pull the container using the following command:

```
docker run --runtime=nvidia --rm -it -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host --cap-add SYS_PTRACE nvcr.io/nvidia/nvtabular:0.1 /bin/bash
```

If you are running on a docker version 19+ please change ```--runtime=nvidia``` to ```--gpus all```.

The container will open a shell when the run command completes execution, you will be responsible for starting the jupyter lab on the docker container.
Should look similar to below:
```
root@2efa5b50b909: 
```

First, activate the ```rapids``` conda environment:
```
root@2efa5b50b909: source activate rapids
```

Then you should see the following prompt (The environment has been activated):
```
(rapids)root@2efa5b50b909: 
```

Finally start the jupyter-lab server:
```
jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='<password>'
```

Now you can use any browser to access the jupyter-lab server, via <MachineIP>:8888

Once in the server, navigate to the ```/nvtabular/``` directory and explore the code base or try out some of the examples. 


Within the container is the codebase, along with all of our dependencies, particularly [RAPIDS cuDF](https://github.com/rapidsai/cudf), and a range of [examples](./examples). The easiest way to get started is to simply launch the container above and explore the examples within.

The code base with examples, can be found at the following directory location within the container:
```
/nvtabular/
```

#### Conda

NVTabular can be installed with Anaconda from the ```nvidia``` channel:

```
conda install -c nvidia -c rapidsai -c numba -c conda-forge nvtabular python=3.6 cudatoolkit=10.2
```

### Examples and Tutorials

An example demonstrating how to use NVTabular to preprocess the [Criteo 1TB dataset](http://labs.criteo.com/2013/12/download-terabyte-click-logs/) can be found in the [criteo example notebook](examples/criteo-example.ipynb). This example also shows how to use NVTabular's data-loaders on the preprocessed data to train Facebook's [Deep Learning Recommender Model (DLRM)](https://github.com/facebookresearch/dlrm/).

Performance of the Criteo DRLM workflow demonstrates the effectiveness of the NVTabular library. The original ETL script provided in Numpy took over five days to complete. Combined with CPU training the total iteration time is over one week. By optimizing the ETL code in spark and running on a DGX-1 equivalent cluster we were able to bring that time down to three hours for ETL and one hour for training.

With NVTabular on a single V100 32GB GPU we are able to complete ETL in 15 minutes, and using a DGX-1 cluster of eight V100 GPUs we can accelerate ETL to 3 minutes. Combined with [HugeCTR](http://www.github.com/NVIDIA/HugeCTR/) we can process the dataset and train the full model in only 18 minutes. This fast iteration is the goal of NVTabular and the [Merlin application framework](https://developer.nvidia.com/nvidia-merlin).


<p align="center"><img src="_images/nvt_performance.png" width="50%"></p>

When examining the relative time spent in ETL vs Training we see that with NVTabular data scientists and ML engineers no longer need to spend 75% of their time on ETL. 

<p align="center"><img src="_images/nvt_relativetime.png" width="50%"></p>

We also have a [simple tutorial](examples/rossmann-store-sales-example.ipynb) that demonstrates similar functionality on a much smaller dataset, providing a pipeline for the [Rossman store sales dataset](https://www.kaggle.com/c/rossmann-store-sales) fed into a [fast.ai tabular data model](https://docs.fast.ai/tabular.html).

### Contributing

If you wish to contribute to the library directly please see [Contributing.md](./CONTRIBUTING.md). We are in particular interested in contributions or feature requests for feature engineering or preprocessing operations that you have found helpful in your own workflows.

To be clear, this is an early alpha release, and we have a long way to go. We have a working framework, but our [operator set](./Operators.md) is limited for the initial launch and every day we are developing new optimizations that will help improve the performance of the library. If you are interested in working with us to help develop this library we are looking for early collaborators and contributors. In the coming months we will be optimizing existing operations, adding a full set of common feature engineering and preprocessing operations, and extending our backend to support multi-node and multi-gpu systems. Please reach out by submitting an issue or see our guide on contributions. We are particularly interested in contributions or feature requests for feature engineering or preprocessing operations that you have found helpful in your own workflows.

### Learn More

If you are interested in learning more about how NVTabular works under the hood we have provided this [more detailed description of the core functionality](HowItWorks.md).

We also have [API documentation](https://nvidia.github.io/NVTabular) that outlines in detail the specifics of the calls available within the library.
