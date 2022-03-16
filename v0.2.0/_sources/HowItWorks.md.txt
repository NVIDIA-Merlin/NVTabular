How it Works
============

![NVTabular Workflow](./images/nvt_workflow.png)

With the transition to v0.2 the NVTabular engine uses the [RAPIDS](http://www.rapids.ai) [Dask-cuDF library](https://github.com/rapidsai/dask-cuda) which provides the bulk of the functionality, accelerating dataframe operations on the GPU, and scaling across multiple GPUs.  NVTabular provides functionality commonly found in deep learning recommendation workflows, allowing you to focus on what you want to do with your data, not how you need to do it.  We also provide a template for our core compute mechanism, Operations, or ‘ops’ allowing you to build your own custom ops from cuDF and other libraries.

Follow our [getting started guide](https://nvidia.github.io/NVTabular/main/Introduction.html#getting-started) to get NVTabular installed on your container or system.  Once installed you can setup a workflow in the following way:

```python
import nvtabular as nvt
workflow = nvt.Workflow(
    cat_names=["user_id", "item_id", "city"],
    cont_names=["age", "time_of_day", "item_num_views"],
    label_name=["label"]
)
```

With the workflow in place we can now explore the library in detail.

Operations
----------
Operations are a reflection of the way in which compute happens on the GPU across large datasets.  At a high level we’re concerned with two types of compute: the type that touches the entire dataset (or some large chunk of it) and the type that operates on a single row.  Operations split the compute such that the first phase, which we call statistics gathering, is the only place where operations that cross the row boundary can take place.  An example of this would be in the Normalize op which relies on two statistics, the mean and standard deviation.  In order to normalize a row, we must first have calculated these two values, and we use a Dask-cudf graph to compute this part of the op.

The second phase of operations is the apply phase, which uses the statistics created earlier to modify the dataset, transforming the data.  Notably we allow for the application of transforms not only during the modification of the dataset, but also during dataloading, with plans to support the same transforms during inference.

```python
# by default, the op will be applied to _all_
# columns of the associated variable type
workflow.add_cont_preprocess(nvt.ops.Normalize())

dataset = nvt.Dataset("/path/to/data.parquet")

# record stats, transform the dataset, and export
# the transformed data to a parquet file
proc.apply(dataset, shuffle=nvt.io.Shuffle.PER_WORKER, output_path="/path/to/export/dir")
```

Dask-cuDF does the scheduling to help optimize the task graph providing an optimal solution to whatever configuration of GPUs you have, from a single GPU to a cluster of many.

A higher level of abstraction
----------------------
NVTabular code is targeted at the operator level, not the dataframe level, providing a method for specifying the operation you want to perform, and the columns or type of data that you want to perform it on.

We make an explicit distinction between feature engineering ops, which create new variables, and preprocessing ops which transform data more directly to make it ready for the model to which it’s feeding.  While the type of computation involved in these two stages is often similar, we want to allow for the creation of new features that will then be preprocessed in the same way as other input variables.

Two main data types are currently supported: categorical variables and continuous variables.  Feature engineering operators explicitly take as input one or more continuous or categorical columns and produce one or more columns of a specific type.  By default the input columns used to create the new feature are also included in the output, however this can be overridden with the [replace] keyword in the operator.  We hope to extend this to multi-hot categoricals in the future in addition to providing specific functionality for high cardinality categoricals which must be treated differently due to memory constraints.

Preprocessing operators take in a set of columns of the same type and perform the operation across each column, transforming the output during the final operation into a long tensor in the case of categorical variables or a float tensor in the case of continuous variables.  Preprocessing operations replace the column values with their new representation by default, but again we allow the user to override this.

```python
# same example as before, but now only apply normalization
# to `age` and `item_num_views` columns, which will create
# new columns `age_normalize` and `item_num_views_normalize`
workflow.add_cont_preprocess(nvt.ops.Normalize(columns=["age", "item_num_views"], replace=False))

dataset = nvt.Dataset("/path/to/data.parquet")
proc.apply(dataset, shuffle=nvt.io.Shuffle.PER_WORKER, output_path="/path/to/export/dir")
```

Operators may also be chained to allow for more complex feature engineering or preprocessing.  Chaining of operators is done by creating a list of the operators.  By default only the final operator in a chain that includes preprocessing will be included in the output with all other intermediate steps implicitly dropped.

```python
# Replace negative and missing values with 0 and then take log(1+x)
workflow.add_cont_feature([FillMissing(), Clip(min_value=0), LogOp()])

# then normalize
workflow.add_cont_preprocess(Normalize())
```

Framework Interoperability
-----------------------

In addition to providing mechanisms for transforming the data to prepare it for deep learning models we also provide framework-specific dataloaders to help optimize getting that data to the GPU.  Under a traditional dataloading scheme, data is read in item by item and collated into a batch.  PyTorch allows for multiple processes to create many batches at the same time, however this still leads to many individual rows of tabular data accessed independently which impacts I/O, especially when this data is on the disk and not in CPU memory.  TensorFlow loads and shuffles TFRecords by adopting a windowed buffering scheme that loads data sequentially to a buffer, from which it randomly samples batches and replenishes with the next sequential elements from disk. Larger buffer sizes ensure more randomness, but can quickly bottleneck performance as TensorFlow tries to keep the buffer saturated. Smaller buffer sizes mean that datasets which aren't uniformly distributed on disk lead to biased sampling and potentially degraded convergence.  

In NVTabular we provide an option to shuffle during dataset creation, creating a uniformly shuffled dataset allowing the dataloader to read in contiguous chunks of data that are already randomized across the entire dataset.  NVTabular provides the option to control the number of chunks that are combined into a batch, allowing the end user flexibility when trading off between performance and true randomization.  This mechanism is critical when dealing with datasets that exceed CPU memory and per epoch shuffling is desired during training.  Full shuffle of such a dataset can exceed training time for the epoch by several orders of magnitude.

Stay tuned for benchmarks on our dataloader performance as compared to those native to the frameworks.

Multi-GPU Support
-----------------------

NVTabular supports multi-GPU scaling with [Dask-CUDA](https://github.com/rapidsai/dask-cuda) and [dask.distributed](https://distributed.dask.org/en/latest/).  To enable distributed parallelism, the NVTabular `Workflow` must be initialized with a `dask.distributed.Client` object:

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

There are currenly many ways to deploy a "cluster" for Dask.  [This article](https://blog.dask.org/2020/07/23/current-state-of-distributed-dask-clusters) gives a nice summary of all practical options.  For a single machine with multiple GPUs, the `dask_cuda.LocalCUDACluster` API is typically the most convenient option.

Since NVTabular already uses [Dask-CuDF](https://docs.rapids.ai/api/cudf/stable/dask-cudf.html) for internal data processing, there are no other requirements for multi-GPU scaling.  With that said, the parallel performance can depend strongly on (1) the size of `Dataset` partitions, (2) the shuffling procedure used for data output, and (3) the specific arguments used for both global-statistics and transformation operations. See the [Multi-GPU](./examples/multigpu) section of this documentation for a simple step-by-step example.

Users are also encouraged to experiment with the [multi-GPU Criteo/DLRM benchmark example](https://github.com/NVIDIA/NVTabular/blob/main/examples/dask-nvtabular-criteo-benchmark.py). For detailed notes on the parameter space for the benchmark, see the [Multi-GPU Criteo Benchmark](./examples/multigpu_bench.md) section of this documentation.

CPU Support
------------
Operators will also be developed using pandas to provide support for users who don’t have access to GPU resources and who wish to use the higher level API that NVTabular provides.  We will try to provide support and feature parity for CPU but GPU acceleration is the focus of this library.  Check the API documentation for coverage.
