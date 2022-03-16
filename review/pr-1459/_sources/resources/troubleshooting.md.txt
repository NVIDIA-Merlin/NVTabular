Troubleshooting
===============

## Checking the Schema of the Parquet File

NVTabular expects that all input parquet files have the same schema, which includes column types and the nullable (not null) option. If you encounter the error
```
RuntimeError: Schemas are inconsistent, try using to_parquet(..., schema="infer"),
or pass an explicit pyarrow schema. Such as to_parquet(..., schema={"column1": pa.string()})
```
when you load the dataset as shown below, one of your parquet files might have a different schema:

```python
ds = nvt.Dataset(PATH, engine="parquet", part_size="1000MB")
ds.to_ddf().head()
```

The easiest way to fix this is to load your dataset with dask_cudf and save it again using the parquet format ( ```dask_cudf.read_parquet("INPUT_FOLDER").to_parquet("OUTPUT_FOLDER")```), so that the parquet file is standardized and the ```_metadata``` file is generated.

If you want to identify which parquet files contain columns with different schemas, you can run one of these scripts:
* [PyArrow](https://github.com/dask/dask/issues/6504#issuecomment-675465645)
* [cudf=0.17](https://github.com/rapidsai/cudf/pull/6796#issue-522934284)

These scripts check for schema consistency and generate only the ```_metadata``` file instead of
converting all the parquet files. If the schema is inconsistent across all files, the script will
raise an exception. For additional information, see [this
issue](https://github.com/NVIDIA/NVTabular/issues/429).

## Reducing Memory Consumption for NVTabular Workflows

NVTabular is designed to scale to larger than GPU or host memory datasets. In our experiments, we are able to [scale to 1.3TB of uncompressed click logs](https://github.com/NVIDIA/NVTabular/tree/main/examples/scaling-criteo). However, some workflows can result in OOM errors `cudaErrorMemoryAllocation out of memory`, which can be addressed by small configuration changes.

### 1. Setting the Row Group Size for the Parquet Files

You can use most Data Frame frameworks to set the row group size (number of rows) for your parquet files. In the following Pandas and cuDF examples, the ```row_group_size``` is the number of rows that will be stored in each row group (internal structure within the parquet file):
```python
#Pandas
pandas_df.to_parquet("/file/path", engine="pyarrow", row_group_size=10000)
#cuDF
cudf_df.to_parquet("/file/path", engine="pyarrow", row_group_size=10000)
```

The row group **memory** size of the parquet files should be smaller than the **part_size** that
you set for the NVTabular dataset such as ```nvt.Dataset(TRAIN_DIR, engine="parquet",
part_size="1000MB")```. To determine how much memory a row group will hold, you can slice your dataframe to a specific number of rows and use the following function to get the memory usage in bytes. You can then set the row_group_size (number of rows) accordingly when you save the parquet file. A row group memory size that is close to 128MB is recommended.

```python
def _memory_usage(df):
    """this function is a workaround for obtaining memory usage lists
    in cudf0.16. This can be deleted and replaced with `df.memory_usage(deep= True, index=True).sum()`
    when using cudf 0.17, which has been fixed as noted on https://github.com/rapidsai/cudf/pull/6549)"""
    size = 0
    for col in df._data.columns:
        if cudf.api.types.is_list_dtype(col.dtype):
            for child in col.base_children:
                size += child.__sizeof__()
        else:
            size += col._memory_usage(deep=True)
    size += df.index.memory_usage(deep=True)
    return size
```

### 2. Initializing a Dask CUDA Cluster

Even if you only have a single GPU to work with, it is best practice to use a distributed Dask-CUDA cluster to execute memory-intensive NVTabular workflows. If there is no distributed `client` object passed to an NVTabular `Workflow`, it will fall back on Dask’s single-threaded “synchronous” scheduler at computation time. The primary advantage of using a Dask-CUDA cluster is that the Dask-CUDA workers enable GPU-aware memory spilling.  In our experience, many OOM errors can be resolved by initializing a dask-CUDA cluster with an appropriate `device_memory_limit` setting, and by passing a corresponding client to NVTabular.  It is easy to deploy a single-machine dask-CUDA cluster using `LocalCUDACluster`.

```python
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

import nvtabular as nvt

cluster = LocalCUDACluster(
    n_workers=1,                        # Number of GPU workers
    device_memory_limit="24GB",         # GPU->CPU spill threshold (~75% of GPU memory)
    rmm_pool_size="28GB",               # Memory pool size on each worker
    local_directory="/nvme/scratch/",   # Fast directory for disk spilling
)
client = Client(cluster)

features = ['col'] >> nvt.ops.Normalize()

workflow = nvt.Workflow(features, client=client)

client.shutdown()
client.close()
```

###  3. String Column Error

If you run into a problem an error that states the size of your string column is too large, like:
```
Exception: RuntimeError('cuDF failure at: /opt/conda/envs/rapids/conda-bld/libcudf_1618503955512/work/cpp/src/copying/concatenate.cu:368: Total number of concatenated rows exceeds size_type range')
```
This is usually caused by string columns in parquet files. If you encounter this error, to fix it you need to decrease the size of the partitions of your dataset. If, after decreasing the size of the partitions, you get a warning about picking a partition size smaller than the row group size, you will need to reformat the dataset with a smaller row group size (refer to #1). There is a 2GB max size for concatenated string columns in cudf currently, for details refer to [this](https://github.com/rapidsai/cudf/issues/3958).   
