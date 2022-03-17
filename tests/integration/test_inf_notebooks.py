#
# Copyright (c) 2021, NVIDIA CORPORATION.
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

import os
from os.path import dirname, realpath

import pytest
from common.utils import _run_notebook

pytest.importorskip("tritonclient")


DATA_DIR = os.environ.get("DATASET_DIR", "/raid/data/")
TEST_PATH = dirname(dirname(realpath(__file__)))

INFERENCE_BASE_DIR = "/model/"
INFERENCE_MULTI_HOT = os.path.join(INFERENCE_BASE_DIR, "models/")

CRITEO_DIR = "examples/scaling-criteo"
ROSSMAN_DIR = "examples/tabular-data-rossmann"
MOVIELENS_DIR = "examples/getting-started-movielens"


@pytest.mark.skip(reason="need to have triton exports already")
def test_inference(asv_db, bench_info, tmpdir, devices):
    # Tritonclient required for this test

    # data_path = os.path.join(INFERENCE_BASE_DIR, "data/")
    input_path = os.path.join(INFERENCE_BASE_DIR, "data/")
    output_path = os.path.join(INFERENCE_BASE_DIR, "data/output")

    os.environ["MODEL_BASE_DIR"] = INFERENCE_MULTI_HOT

    # Run Criteo inference
    notebook = os.path.join(
        dirname(TEST_PATH), CRITEO_DIR, "04-Triton-Inference-with-HugeCTR.ipynb"
    )
    _run_notebook(tmpdir, notebook, input_path, output_path, gpu_id=devices, clean_up=False)

    notebook = os.path.join(dirname(TEST_PATH), CRITEO_DIR, "04-Triton-Inference-with-TF.ipynb")
    _run_notebook(tmpdir, notebook, input_path, output_path, gpu_id=devices, clean_up=False)

    # Run Movielens inference
    notebook = os.path.join(
        dirname(TEST_PATH), MOVIELENS_DIR, "inference-HugeCTR/Triton-Inference-with-HugeCTR.ipynb"
    )
    _run_notebook(tmpdir, notebook, input_path, output_path, gpu_id=devices, clean_up=False)

    notebook = os.path.join(dirname(TEST_PATH), MOVIELENS_DIR, "04-Triton-Inference-with-TF.ipynb")
    _run_notebook(tmpdir, notebook, input_path, output_path, gpu_id=devices, clean_up=False)
