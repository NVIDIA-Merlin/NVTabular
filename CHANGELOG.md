
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
