#!/bin/bash

# Call this script with name of container as parameter
# [merlin-training, merlin-tensorflow-training,
#  merlin-pytorch-training, merlin-inference]

# Get last NVTabular version
cd /nvtabular/
git pull origin main

# Run tests for all containers
pytest tests/integration/test_notebooks.py::test_criteo_example
pytest tests/integration/test_notebooks.py::test_rossman_example
pytest tests/integration/test_notebooks.py::test_movielens_example

# Run tests for specific containers
if [ "$1" == "merlin-training" ]; then
  pytest tests/integration/test_nvt_hugectr.py::test_training
elif [ "$1" == "merlin-tensorflow-trainig" ]; then
  pytest tests/integration/test_nvt_tf_inference.py::test_nvt_tf_rossmann_inference
  pytest tests/integration/test_nvt_tf_inference.py::test_nvt_tf_movielens_inference
elif [ "$1" == "merlin-pytorch-training" ]; then
  echo "Nothing specific for $1 yet"
elif [ "$1" == "merlin-inference" ]; then
  pytest tests/integration/test_nvt_tf_inference.py::test_nvt_tf_rossmann_inference_triton
  pytest tests/integration/test_nvt_tf_inference.py::test_nvt_tf_rossmann_inference_triton_mt
  pytest tests/integration/test_nvt_tf_inference.py::test_nvt_tf_movielens_inference_triton
  pytest tests/integration/test_nvt_tf_inference.py::test_nvt_tf_movielens_inference_triton_mt
  pytest tests/integration/test_nvt_hugectr.py::test_inference
else
  echo "INVALID Container name"
  exit 1
fi

