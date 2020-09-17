# Multi-GPU Criteo/DLRM Benchmark Overview

The benchmark script described in this document is located at `NVTabular/examples/dask-nvtabular-criteo-benchmark.py`.

The [multi-GPU Criteo/DLRM benchmark](https://github.com/NVIDIA/NVTabular/blob/main/examples/dask-nvtabular-criteo-benchmark.py) is designed to measure the time required to preprocess the [Criteo (1TB) dataset](https://www.kaggle.com/c/criteo-display-ad-challenge/data) for Facebook’s [DLRM model](https://github.com/facebookresearch/dlrm).  The user must specify the path of the raw dataset (using the `--data-path` flag), as well as the output directory for all temporary/final data (using the `--out-path` flag).

### Example Usage


```python
python dask-nvtabular-criteo-benchmark.py --data-path /path/to/criteo_parquet --out-path /out/dir/`
```

### Dataset Requirements (Parquet)


The script is designed with a parquet-formatted dataset in mind. Although csv files can also be handled by NVTabular, converting to parquet yields significantly better performance.  To convert your dataset, try using the [conversion notebook](https://github.com/NVIDIA/NVTabular/blob/main/examples/optimize_criteo.ipynb) (located at `NVTabular/examples/optimize_criteo.ipynb`).

### General Notes on Parameter Tuning

The script was originally developed and tested on an NVIDIA DGX-1 machine (8x 32GB V100s, 1TB RAM).  Users with limited device and/or host memory may experience memory errors with the default options. Depending on the system, these users may need to modify one or more of the “Algorithm Options” described below. For example, it may be necessary to expand the list of “high-cardinality” columns, increase the tree-width and/or use “disk” for the cat-cache options.

In addition to adjusting the algorithm details, users with limited device memory may also find it useful to adjust the `--device-pool-frac` and/or `--device-limit-frac` options (reduce both fractions).

For all users, the most important benchmark options include the following.

- **Device list**: `-d` (easiest way to set the number of workers)
- **Partition size**: `—part-mem-frac` (bigger partitions = better device efficiency)
- **Intermediate-category storage**: `—cats-on-device` (always use this option when total device memory is sufficiently large)
- **Communication protocol**: `-p` (use ucx whenever possible)
- **Output-file count**: `—out-files-per-proc` (fewer output files is faster)

See option descriptions below for more information.


## Parameter Overview

### System Options


##### Visible Devices
By default, the script will deploy a cluster with a single Dask-CUDA worker on  every GPU specified by the `CUDA_VISIBLE_DEVICES` environment variable. The user may also specify distinct list of devices using the `-d` flag.

e.g. `-d 1,2,3`


##### Communication Protocol
By default, the Dask-CUDA cluster will use a “tcp” protocol for inter-process communication.  Users may also elect to use “ucx” to take advantage of NVLink and/or Infiniband technologies.  The “ucx” option is highly recommended, but requires ucx-py to be installed.

e.g. `-p ucx`


##### Memory Management
By default, the Dask-CUDA workers will use an RMM memory pool to avoid memory-allocation bottlenecks, and will spill data from device to host memory when Dask-aware usage exceeds a specific threshold.  The size of the RMM memory pool On each worker defaults to 90% of the total capacity, but the user can specify a different fraction using the `--device-pool-frac` flag.  If `0` is specified, no memory pool will be used.

e.g. `--device-pool-frac 0.5`

By default, the Dask-CUDA workers will begin spilling data from device memory to host memory when the input/output data of in-memory tasks exceeds 80% of the total capacity.  For systems with limited device memory, temporary allocation made during task execution may still lead to out-of-memory (OOM) errors.  To modify the threshold, the user can specify a different fraction Using the `--device-limit-frac` flag.

e.g. `--device-limit-frac 0.66`

##### IO Threads (Writing)
By default, multi-threading will not be used to write output data. Some systems may see better performance when 2+ threads are used to overlap sequencial writes by the same worker. The user can specify a specific number of threads using the `--num-io-threads` flag.

e.g. `--num-io-threads 2`

Note that multi-threading may reduce the optimal partition size (see the `--part-mem-frac` flag below).

### Data-Decomposition Options


##### Partition Sizes (dataset chunking)
To process out-of-core data, NVTabular uses Dask-CuDF to partition the data into a lazily-evaluated collection of CuDF DataFrame objects.  By default the maximum size of these so-called partitions will be approximately 12.5% of a single-GPUs memory capacity.  The user can modify the desired size of partitions by passing a fractional value with the `--part-mem-frac` flag.

e.g. `--part-mem-frac 0.16`

##### Output-File Count
NVTabular uses the file structure of the output dataset to shuffle data as it is written to disk.  That is, after a Dask-CUDA worker transforms a dataset partition, it will append (random) splits of that partition to some number of output files.  Each worker will manage its own set of output files.  The `--out-files-per-proc` can be used to modify the number of output files managed by each worker (defaults to 8).  Since output files are uniquely mapped to processes, the total number of output files is multiplied by the number of workers.

e.g. `--out-files-per-proc 24`

Note that a large number of output files may be required to perform the “PER_WORKER” shuffle option (see description of the `—shuffle` flag below).  This is because each file will be fully shuffled in device memory.

##### Shuffling
NVTabular currently offers two options for shuffling output data to disk. The `“PER_PARTITION”` option means that each partition will be independently shuffled after transformation, and then appended to some number of distinct output files.  The number of output files is specified by the `--out-files-per-proc` flag (described above), and the files  are uniquely mapped to each worker.  The `“PER_WORKER”` option follows the same process, but the “files” are initially written to in-host-memory, and then reshuffled and persisted to disk after the full dataset is processed.  The user can specify the specific shuffling algorithm to use with the `—shuffle` flag.

e.g. `—shuffle PER_WORKER`

Note that the `“PER_WORKER”` option may require a larger-than default output-file count (See description of the `--out-files-per-proc` flag above).


### Preprocessing Options

##### Column Names
By default this script will assume the following categorical and continuous column names.

- Categorical: “C1”, ”C2”, … , ”C26”
- Continuous: “I1”, ”I2”, … , ”I13”

The user may specify different column names, or a subset of these names, by passing a column-separated list to  the `--cat-names` and/or `—cont_names` flags.

e.g. `—cat-names C01,C02  --cont_names I01,I02 —high_cards C01`

Note that, if your dataset includes non-default column names, you should also use the `—high-cards` flag (described below), to specify the names of high-cardinality columns.

##### Categorical Encoding
By default, all categorical-column groups will be used for the final encoding transformation.  The user can also specify a frequency threshold for groups to be included in the encoding with the `—freq-limit` (or `-f`) flag.

e.g `-f 15` (groups with fewer than 15 instances in the dataset will not be used for encoding)
### Algorithm Options

##### High-Cardinality Columns

As described below, the specific algorithm used for categorical encoding can be column dependent.  In this script, we use special options for a subset of “high-cardinality” columns.  By default, these columns are "C20,C1,C22,C10”.  However, the user can specify different column names with the `--high-cards` flag.

e.g. `—high_cards C01,C10`

Note that only the columns specified with this flag (or the default columns) will be targetedby the `--tree-width` and/or `--cat-cache-high` flags (described below).

##### Global-Categories Calculation (GroupbyStatistics)
In order encode categorical columns, NVTabular needs to calculate a global list of unique categories for each categorical column. This is accomplished with a global groupby-aggregation-based tree reduction on each column. In order to avoid memory pressure on the device, intermediate groupby data is moved to host memory between tasks in the global-aggregation tree.  For users with a sufficient amount of total GPU memory, this device-to-host transfer can be avoided with the by adding the `—cats-on-device` flag to the execution command.

e.g. `—cats-on-device`

In addition to controlling device-to-host data movement, the user can also use the `--tree-width` flag to specify the width of the groupby-aggregation tree for high-cardinality columns.  Although NVTabular allows the user to specify the tree-width for each column independently, this option will target all columns  specified with `—high-cards`.

e.g. `—tree_width 4`


##### Categorical Encoding (Categorify)
During the categorical-encoding transformation stage, the column-specific unique values must be read into GPU memory for the operation.  Since each NVTabular process will only operate on a single partition at a time, the same unique-value statistics need to be re-read (for every categorical column) for each partition that is transformed.  Unsurprisingly, the performance of categorical encoding can be dramatically improved by caching the unique values on each worker between transformation operations.

The user can specify caching location for low- and high-cardinality columns separately. Recall that high-cardinality columns can be specified with `—high_cards` (and all remaining categorical columns will be treated as low-cardinality”).  The user can specify the caching location of low-cardinality columns with the `--cat-cache-low` flag, and high-cardinality columns with the `--cat-cache-low` flag.  For both cases, the options are “device”, “host”, or “disk”.

e.g. `--cat-cache-low device  --cat-cache-high host`


### Diagnostics Options


##### Dashboard
A wonderful advantage of the Dask-Distributed ecosystem is the convenient set of diagnostics utilities.  By default (if Bokeh is installed on your system), the distributed scheduler will host a diagnostics dashboard at  `http://localhost:8787/status` (where localhost may need to be changed to the the IP address where the scheduler is running).  If port 8787 is already in use, a different (random) port will be used.  However, the user can specify a specific port using the `—dashboard-port` flag.

e.g. `—dashboard-port 3787`

##### Profile
In addition to hosting a diagnostics dashboard, the distributed cluster can also collect and export profiling data on all scheduler and worker processes.  To export an interactive profile report, the user can specify a file path with the `—profile` flag.  If this flag is not used, no profile will be collected/exported.

e.g. `—profile my-profile.html`