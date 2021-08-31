# NVTabular v0.6.1 (11 August 2021)

## Bug Fixes

* Fix installing package via pip [#1030](https://github.com/NVIDIA/NVTabular/pull/1030)
* Fix inference with groupby operator [#1019](https://github.com/NVIDIA/NVTabular/issues/1019)
* Install tqdm with conda package [#1030](https://github.com/NVIDIA/NVTabular/pull/1030)
* Fix workflow output_dtypes with empty partitions [#1028](https://github.com/NVIDIA/NVTabular/pull/1028)

# NVTabular v0.6.0 (5 August 2021)

## Improvements
* Add CPU support [#534](https://github.com/NVIDIA/NVTabular/issues/534)
* Speed up inference on Triton Inference Server [#744](https://github.com/NVIDIA/NVTabular/issues/744)
* Add support for session based recommenders [#355](https://github.com/NVIDIA/NVTabular/issues/355)
* Add PyTorch Dataloader support for Sparse Tensors [#500](https://github.com/NVIDIA/NVTabular/issues/500)
* Add ListSlice operator for truncating list columns [#734](https://github.com/NVIDIA/NVTabular/issues/734)
* Categorical ids sorted by frequency [#799](https://github.com/NVIDIA/NVTabular/issues/799)
* Add ability to select a subset of a ColumnGroup [#809](https://github.com/NVIDIA/NVTabular/issues/809)
* Add option to use Rename op to give a single column a new fixed name [#825](https://github.com/NVIDIA/NVTabular/issues/824)
* Add a 'map' function to KerasSequenceLoader, which enables sample weights [#667](https://github.com/NVIDIA/NVTabular/issues/667)
* Add JoinExternal option on nvt.Dataset in addition to cudf [#370](https://github.com/NVIDIA/NVTabular/issues/370)
* Allow passing ColumnGroup to get_embedding_sizes [#732](https://github.com/NVIDIA/NVTabular/issues/732)
* Add ability to name LambdaOp and provide a better default name in graph visualizations [#860](https://github.com/NVIDIA/NVTabular/issues/860)

## Bug Fixes

* Fix make_feature_column_workflow for Categorical columns [#763](https://github.com/NVIDIA/NVTabular/issues/763)
* Fix Categorify output dtypes for list columns [#963](https://github.com/NVIDIA/NVTabular/issues/963)
* Fix inference for Outbrain example [#669](https://github.com/NVIDIA/NVTabular/issues/669)
* Fix dask metadata after calling workflow.to_ddf() [#852](https://github.com/NVIDIA/NVTabular/issues/734)
* Fix out of memory errors [#896](https://github.com/NVIDIA/NVTabular/issues/896), [#971](https://github.com/NVIDIA/NVTabular/pull/971)
* Fix normalize output when stdev is zero [#993](https://github.com/NVIDIA/NVTabular/pull/993)
* Fix using UCX with a dask cluster on Merlin containers [#872](https://github.com/NVIDIA/NVTabular/pull/872)

# NVTabular v0.5.3 (1 June 2021)

## Bug Fixes
* Fix Shuffling in Torch DataLoader [#818](https://github.com/NVIDIA/NVTabular/pull/818)
* Fix "Unsupported type_id conversion" in triton inference for string columns [#813](https://github.com/NVIDIA/NVTabular/issues/813)
* Fix HugeCTR inference backend [Merlin#8](https://github.com/NVIDIA-Merlin/Merlin/pull/8)

# NVTabular v0.5.1 (4 May 2021)

## Improvements

* Update dependencies to use cudf 0.19
* Removed conda from docker containers, leading to much smaller container sizes
* Added CUDA 11.2 support
* Added FastAI v2.3 support

## Bug Fixes

* Fix NVTabular preprocessing with HugeCTR inference

# NVTabular v0.5.0 (13 April 2021)

## Improvements

* Adding Horovod integration to NVTabular's dataloaders, allowing you to use multiple GPU's to train TensorFlow and PyTorch models
* Adding a Groupby operation for use with session based recommender models
* Added ability to read and write datasets partitioned by a column, allowing 
* Add example notebooks for using Triton Inference Server with NVTabular
* Restructure and simplify Criteo example notebooks
* Add support for PyTorch inference with Triton Inference Server

## Bug Fixes

* Fix bug with preprocessing categorical columns with NVTabular not working with HugeCTR and Triton Inference Server [#707](https://github.com/NVIDIA/NVTabular/issues/707)

# NVTabular v0.4.0 (9 March 2021)

## Breaking Changes

* The API for NVTabular has been signficantly refactored, and existing code targetting the 0.3 API will need to be updated.
Workflows are now represented as graphs of operations, and applied using a sklearn 'transformers' style api. Read more by
checking out the [examples](https://nvidia.github.io/NVTabular/v0.4.0/examples/index.html)

## Improvements

* Triton integration support for NVTabular with TensorFlow and HugeCTR models
* Recommended cloud configuration and support for AWS and GCP
* Reorganized examples and documentation
* Unified Docker containers for Merlin components (NVTabular, HugeCTR and Triton)
* Dataset analysis and generation tools

# NVTabular v0.3.0 (23 November 2020)

## Improvements

* Add MultiHot categorical support for both preprocessing and dataloading
* Add support for pretrained embeddings to the dataloaders
* Add a Recsys2020 competition example notebook
* Add ability to automatically map tensorflow feature columns to a NVTabular workflow
* Multi-Node support

# NVTabular v0.2.0 (10 September 2020)

## Improvements

* Add Multi-GPU support using Dask-cuDF
* Add support for reading datasets from S3, GCS and HDFS
* Add 11 new operators: ColumnSimilarity, Dropna, Filter, FillMedian, HashBucket, JoinGroupBy, JoinExternal, LambdaOp, NormalizeMinMax, TargetEncoding and DifferenceLag
* Add HugeCTR integration and an example notebook showing an end to end workflow
* Signicantly faster dataloaders featuring a unified backend between TensorFlow and PyTorch

# NVTabular v0.1.1 (3 June 2020)

## Improvements

* Switch to using the release version of cudf 0.14

## Bug Fixes

* Fix PyTorch dataloader for compatability with deep learning examples
* Fix FillMissing operator with constant fill
* Fix missing yaml dependency on conda install
* Fix get_emb_sz off-by-one error

# NVTabular v0.1.0 - (13 May 2020)

* Initial public release of NVTabular
