# NVTabular v0.2.0 (10 September 2020)

## Improvements

* Add Multi-GPU support using Dask-cuDF
* Add support for reading datasets from S3, GCS and HDFS
* Add 11 new operators: ColumnSimilarity, Dropna, Filter, FillMedian, HashBucket, JoinGroupBy, JoinExternal, LambdaOp, NormalizeMinMax, TargetEncoding and DifferenceLag
* Add HugeCTR integration and an example ntoebook showing an end to end workflow
* Signicantly faster dataloaders featured a unified backend between TensorFlow and PyTorch

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
