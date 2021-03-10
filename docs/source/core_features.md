Core Features
=============

NVTabular supports the following core features:
* [TensorFlow and PyTorch Interoperability](#tensorflow-and-pytorch-interoperability)
* [HugeCTR Interoperability](#hugectr-interoperability)
* [Multi-GPU Support](#multi-gpu-support)
* [Multi-Node Support](#multi-node-support)
* [Multi-Hot Encoding and Pre-Existing Embeddings](#multi-hot-encoding-and-pre-existing-embeddings)
* [Shuffling Datasets](#shuffling-datasets)
* [Cloud Integration](#cloud-integration)

## TensorFlow and PyTorch Interoperability ##

In addition to providing mechanisms for transforming the data to prepare it for deep learning models, we also have framework-specific dataloaders implemented to help optimize getting that data to the GPU. Under a traditional dataloading scheme, data is read item by item and collated into a batch. With PyTorch, multiple processes can create many batches at the same time. However, this still leads to many individual rows of tabular data being accessed independently, which impacts I/O, especially when this data is on the disk and not in the CPU memory. TensorFlow loads and shuffles TFRecords by adopting a windowed buffering scheme that loads data sequentially to a buffer, which it randomly samples batches and replenishes with the next sequential elements from the disk. Larger buffer sizes ensure more randomness, but can quickly bottleneck performance as TensorFlow tries to keep the buffer saturated. Smaller buffer sizes mean that datasets, which aren't uniformly distributed on the disk, lead to biased sampling and potentially degraded convergence.

## HugeCTR Interoperability ##

NVTabular is also capable of preprocessing datasets that can be passed to HugeCTR for training. For additional information, see the [HugeCTR Example Notebook](https://nvidia.github.io/NVTabular/main/examples/scaling-criteo/02-03c-ETL-with-NVTabular-HugeCTR.html) for details about how this works.

## Multi-GPU Support ##

NVTabular supports multi-GPU scaling with [Dask-CUDA](https://github.com/rapidsai/dask-cuda) and [dask.distributed](https://distributed.dask.org/en/latest/). To enable distributed parallelism, the NVTabular `Workflow` must be initialized with a `dask.distributed.Client` object as follows:

```python
import nvtabular as nvt
from dask.distributed import Client

# Deploy a new cluster
# (or specify the port of an existing scheduler)
cluster = "tcp://MachineA:8786"

client = Client(cluster)
workflow = nvt.Workflow(..., client=client)
...
```

Currently, there are many ways to deploy a "cluster" for Dask. This [article](https://blog.dask.org/2020/07/23/current-state-of-distributed-dask-clusters) gives a summary of all the practical options. For a single machine with multiple GPUs, the `dask_cuda.LocalCUDACluster` API is typically the most convenient option.

Since NVTabular already uses [Dask-CuDF](https://docs.rapids.ai/api/cudf/stable/dask-cudf.html) for internal data processing, there are no other requirements for multi-GPU scaling. With that said, the parallel performance can depend strongly on (1) the size of `Dataset` partitions, (2) the shuffling procedure used for data output, and (3) the specific arguments used for both global-statistics and transformation operations. For additional information, see [Multi-GPU](https://github.com/NVIDIA/NVTabular/blob/main/examples/multi-gpu-toy-example/multi-gpu_dask.ipynb) for a simple step-by-step example.

We encourage experimentation with the [multi-GPU Criteo/DLRM benchmark example](https://github.com/NVIDIA/NVTabular/blob/main/examples/scaling-criteo/dlrm_fp32_64k.json).

## Multi-Node Support ##

NVTabular supports multi-node scaling with [Dask-CUDA](https://github.com/rapidsai/dask-cuda) and [dask.distributed](https://distributed.dask.org/en/latest/). To enable distributed parallelism, start a cluster and connect to it to run the application by doing the following:

1) Start the scheduler `dask-scheduler`.
2) Start the workers `dask-cuda-worker schedulerIP:schedulerPort`.
3) Run the NVTabular application where the NVTabular `Workflow` has been initialized as described in the Multi-GPU Support section.

For a detailed description of each existing method that is needed to start a cluster, please read this [article](https://blog.dask.org/2020/07/23/current-state-of-distributed-dask-clusters).

## Multi-Hot Encoding and Pre-Existing Embeddings ##

NVTabular supports the:

* processing of datasets with multi-hot categorical columns.
* passing of continuous vector features like pre-trained embeddings, which includes basic preprocessing and feature engineering, as well as full support in the dataloaders for training models with both TensorFlow and PyTorch.

Multi-hot lets you represent a set of categories as a single feature. For example, in a movie recommendation system, each movie might have a list of genres associated with it like comedy, drama, horror, or science fiction. Since movies can belong to more than one genre, we can't use single-hot encoding like we are doing for scalar
columns. Instead we train models with multi-hot embeddings for these features by having the deep learning model look up an embedding for each category in the list and then average all the embeddings for each row. Both multi-hot categoricals and vector continuous features are represented using list columns in our datasets. cuDF has recently added support for list columns, and we're leveraging that support in NVTabular to power this feature. 

Our Categorify and HashBucket operators can map list columns down to small contiguous integers, which are suitable for use in an embedding lookup table. This is only possible if the dataset contains two rows like ```[['comedy', 'horror'], ['comedy', 'sciencefiction']]``` so that NVTabular can transform the strings for each row into categorical IDs like ```[[0, 1], [0, 2]]``` to be used in our embedding layers.

Our PyTorch and TensorFlow dataloaders have been extended to handle both categorical and continuous list columns. In TensorFlow, the KerasSequenceLoader class will transform each list column into two tensors representing the values and offsets into those values for each batch. These tensors can be converted into RaggedTensors for multi-hot columns, and for vector continuous columns where the offsets tensor can be safely ignored. We've provided a ```nvtabular.framework_utils.tensorflow.layers.DenseFeatures``` Keras layer that will automatically handle these conversions for both continuous and categorical columns. For PyTorch, there's support for multi-hot columns to our ```nvtabular.framework_utils.torch.models.Model``` class, which internally is using the PyTorch [EmbeddingBag](https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html) layer to handle the multi-hot columns.

## Shuffling Datasets ##

NVTabular makes it possible to shuffle during dataset creation. This creates a uniformly shuffled dataset that allows the dataloader to load large contiguous chunks of data, which are already randomized across the entire dataset. NVTabular also makes it possible to control the number of chunks that are combined into a batch, providing flexibility when trading off between performance and true randomization. This mechanism is critical when dealing with datasets that exceed CPU memory and individual epoch shuffling is desired during training. Full shuffle of such a dataset can exceed training time for the epoch by several orders of magnitude.

## Cloud Integration ##

NVTabular offers cloud integration with Amazon Web Services (AWS) and Google Cloud Platform (GCP), giving you the ability to build, train, and deploy models on the cloud using datasets. For additional information, see [Amazon Web Services](./resources/cloud_integration.md#amazon-web-services) and [Google Cloud Platform](./resources/cloud_integration.md#google-cloud-platform).
