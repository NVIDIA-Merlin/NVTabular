#
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#!/bin/bash
set -e

# Call this script with:
# 1. Name of container as first parameter
#    [merlin-hugectr,  merlin-tensorflow,  merlin-pytorch]
#
# 2. Devices to use:
#    [0; 0,1; 0,1,..,n-1]

cd /nvtabular/

container=$1
config="-rsx --devices $2"

# Run tests for training containers
#pytest $config tests/integration/test_notebooks.py::test_criteo
pytest $config tests/integration/test_notebooks.py::test_rossman
pytest $config tests/integration/test_notebooks.py::test_movielens

# Run tests for specific containers
if [ "$container" == "merlin-hugectr" ]; then
  pytest $config tests/integration/test_nvt_hugectr.py::test_training
  # pytest $config tests/integration/test_notebooks.py::test_criteo
  # pytest $config tests/integration/test_nvt_hugectr.py::test_inference
elif [ "$container" == "merlin-tensorflow" ]; then
  pytest $config tests/integration/test_nvt_tf_inference.py::test_nvt_tf_rossmann_inference
  pytest $config tests/integration/test_nvt_tf_inference.py::test_nvt_tf_movielens_inference
  # pytest $config tests/integration/test_nvt_tf_inference.py::test_nvt_tf_rossmann_inference_triton
  # pytest $config tests/integration/test_nvt_tf_inference.py::test_nvt_tf_rossmann_inference_triton_mt
  # pytest $config tests/integration/test_nvt_tf_inference.py::test_nvt_tf_movielens_inference_triton
  # pytest $config tests/integration/test_nvt_tf_inference.py::test_nvt_tf_movielens_inference_triton_mt
fi
