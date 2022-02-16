#!/bin/bash
set -e

# Call this script with:
# 1. Name of container as first parameter
#    [merlin-training, merlin-tensorflow-training,
#      merlin-pytorch-training, merlin-inference]
#
# 2. Devices to use:
#    [0; 0,1; 0,1,..,n-1]

# Get last NVTabular version
cd /nvtabular/
git pull origin main

container=$1
config="--devices $2"

# Run tests for all containers but inference
if [ "$container" != "merlin-inference" ]; then
  #pytest $config tests/integration/test_notebooks.py::test_criteo
  pytest $config tests/integration/test_notebooks.py::test_rossman
  pytest $config tests/integration/test_notebooks.py::test_movielens
fi

# Run tests for specific containers
if [ "$container" == "merlin-training" ]; then
  pytest $config tests/integration/test_nvt_hugectr.py::test_training
elif [ "$container" == "merlin-tensorflow-training" ]; then
  pytest $config tests/integration/test_nvt_tf_inference.py::test_nvt_tf_rossmann_inference
  pytest $config tests/integration/test_nvt_tf_inference.py::test_nvt_tf_movielens_inference
elif [ "$container" == "merlin-pytorch-training" ]; then
  echo "Nothing specific for $container yet"
elif [ "$container" == "merlin-inference" ]; then
  #pytest $config tests/integration/test_notebooks.py::test_inference
  pytest $config tests/integration/test_nvt_tf_inference.py::test_nvt_tf_rossmann_inference_triton
  pytest $config tests/integration/test_nvt_tf_inference.py::test_nvt_tf_rossmann_inference_triton_mt
  pytest $config tests/integration/test_nvt_tf_inference.py::test_nvt_tf_movielens_inference_triton
  pytest $config tests/integration/test_nvt_tf_inference.py::test_nvt_tf_movielens_inference_triton_mt
  pytest $config tests/integration/test_nvt_hugectr.py::test_inference
else
  echo "INVALID Container name"
  exit 1
fi
