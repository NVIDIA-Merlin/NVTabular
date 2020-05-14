How it Works
============

![NVTabular Workflow](./_images/nvt_workflow.png)

NVTabular wraps the RAPIDS cuDF library which provides the bulk of the functionality, accelerating dataframe operations on the GPU.  We found in our internal usage of cuDF on massive dataset like [Criteo](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) or [RecSys 2020](https://recsys-twitter.com/) that it wasn’t straightforward to use once the dataset had scaled past GPU memory.  The same design pattern kept emerging for us and we decided to package it up as NVTabular in order to make tabular data workflows simpler.

We provide mechanisms for iteration when the dataset exceeds GPU memory, allowing you to focus on what you want to do with your data, not how you need to do it.  We also provide a template for our core compute mechanism, Operations, or ‘ops’ allowing you to build your own custom ops from cuDF and other libraries.

Follow our getting started guide to get NVTabular installed on your container or system.  Once installed you can setup a workflow in the following way:

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
Operations are a reflection of the way in which compute happens on the GPU across large datasets.  At a high level we’re concerned with two types of compute: the type that touches the entire dataset (or some large chunk of it) and the type that operates on a single row.  Operations split the compute such that the first phase, which we call statistics gathering, is the only place where operations that cross the row boundary can take place.  An example of this would be in the Normalize op which relies on two statistics, the mean and standard deviation.  In order to normalize a row, we must first have calculated these two values.

Statistics are further split into a chunked compute and a combine stage allowing for chunked iteration across datasets that don’t fit in GPU (or CPU) memory.  Where possible (and efficient) we utilize the GPU to do highly parallel compute, but many operations also rely on host memory for buffering and CPU compute when necessary.  The chunked results are combined to provide the statistics necessary for the next phase.

The second phase of operations is the apply phase, which uses the statistics created earlier to modify the dataset, transforming the data.  Notably we allow for the application of transforms not only during the modification of the dataset, but also during dataloading, with plans to support the same transforms during inference.

```python
# by default, the op will be applied to _all_
# columns of the associated variable type
workflow.add_cont_preprocess(nvt.ops.Normalize())

dataset = nvt.dataset("/path/to/data.parquet", engine="parquet", gpu_memory_frac=0.2)

# record stats, transform the dataset, and export
# the transformed data to a parquet file
proc.apply(train_ds_iterator, apply_offline=True, record_stats=True, shuffle=True, output_path="/path/to/export/dir")
```

In order to minimize iteration through the data we combine all of the computation required for statistics into a single computation graph that is applied chunkwise while the data is on GPU.  We similarly group the apply operation and transform the entire chunk at a time.  This lazy iteration style allows you to setup a desired workflow first, and then apply it to multiple datasets, including the option to apply statistics from one dataset to others.  Using this option the training set statistics can be applied to the validation and test sets preventing undesirable data leakage.

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

dataset = nvt.dataset("/path/to/data.parquet", engine="parquet", gpu_memory_frac=0.2)
proc.apply(train_ds_iterator, apply_offline=True, record_stats=True, shuffle=True, output_path="/path/to/export/dir")
```

Operators may also be chained to allow for more complex feature engineering or preprocessing.  Chaining of operators is done by creating a list of the operators.  By default only the final operator in a chain that includes preprocessing will be included in the output with all other intermediate steps implicitly dropped.

```python
# zero fill and then take log(1+x)
workflow.add_cont_feature([ZeroFill(), LogOp()])

# then normalize
workflow.add_cont_preprocess(Normalize())
```

Framework Interoperability
-----------------------

In addition to providing mechanisms for transforming the data to prepare it for deep learning models we also provide framework-specific dataloaders to help optimize getting that data to the GPU.  Under a traditional dataloading scheme, data is read in item by item and collated into a batch.  PyTorch allows for multiple processes to create many batches at the same time, however this still leads to many individual rows of tabular data accessed independently which impacts I/O, especially when this data is on the disk and not in CPU memory.  TensorFlow loads and shuffles TFRecords by adopting a windowed buffering scheme that loads data sequentially to a buffer, which it samples batches from randomly and replenishes with the next sequential elements from disk. Larger buffer sizes ensure more randomness, but can quickly bottleneck performance as TensorFlow tries to keep the buffer saturated. Smaller buffer sizes mean that datasets which aren't uniformly distributed on disk lead to biased sampling and potentially degraded convergence.  

In NVTabular we provide an option to shuffle during dataset creation, creating a uniformly shuffled allowing the dataloader to read in contiguous chunks of data that are already randomized across the entire dataset.  NVTabular provides the option to control the number of chunks that are combined into a batch, allowing the end user flexibility when trading off between performance and true randomization.  This mechanism is critical when dealing with datasets that exceed CPU memory and per epoch shuffling is desired during training.  Full shuffle of such a dataset can exceed training time for the epoch by several orders of magnitude.

When compared to an item by item dataloader of PyTorch we have benchmarked our throughput as 100x faster dependent upon batch and tensor size.  Relative to Tensorflow’s windowed shuffle NVTabular is ~2.5x faster with many optimizations still available.

Multi-GPU Support
-----------------------

The next release will have multi-GPU support using Dask-cudf and Dask, and will allow for easy parallelization of operations across multiple GPUs.

CPU Support
------------
Operators will also be developed using pandas to provide support for users who don’t have access to GPU resources and who wish to use the higher level API that NVTabular provides.  We will try to provide support and feature parity for CPU but GPU acceleration is the focus of this library.  Check the API documentation for coverage.
