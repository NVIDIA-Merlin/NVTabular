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
import warnings
from distutils.spawn import find_executable

import cudf
import cupy as cp
import numpy as np
import pytest
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

import nvtabular as nvt
import tests.conftest as test_utils

MODEL_DIR = "~/nvt-examples/models/"
DATA_DIR = "~/nvt-examples/data/"

TRITON_SERVER_PATH = find_executable("tritonserver")


# Update TEST_N_ROWS param in test_nvt_tf_trainin.py to test larger sizes
@pytest.mark.parametrize("n_rows", [64, 45, 32, 14, 7, 1])
@pytest.mark.parametrize("err_tol", [0.00001])
def test_nvt_tf_movielens_inference(n_rows, err_tol):

    INPUT_DATA_DIR = os.path.expanduser("~/nvt-examples/movielens/data/")

    warnings.simplefilter("ignore")

    model_name = "test_movielens_tf"
    col_names = ["userId", "movieId"]
    # read in a batch of data to get transforms for
    batch = cudf.read_csv(os.path.join(INPUT_DATA_DIR, "test_data.csv"), nrows=n_rows)[col_names]

    # convert the batch to a triton inputs
    columns = [(col, batch[col]) for col in col_names]
    inputs = []

    col_dtypes = [np.int64, np.int64]
    for i, (name, col) in enumerate(columns):
        d = col.values_host.astype(col_dtypes[i])
        d = d.reshape(len(d), 1)
        inputs.append(grpcclient.InferInput(name, d.shape, np_to_triton_dtype(col_dtypes[i])))
        inputs[i].set_data_from_numpy(d)

    # placeholder variables for the output
    outputs = []
    outputs.append(grpcclient.InferRequestedOutput("output"))
    # make the request
    with grpcclient.InferenceServerClient("localhost:8001") as client:
        response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    output_actual = cudf.read_csv(os.path.join(INPUT_DATA_DIR, "output.csv"), nrows=n_rows)
    output_actual = cp.asnumpy(output_actual["0"].values)
    output_predict = response.as_numpy("output")

    diff = abs(output_actual - output_predict[:, 0])
    assert (diff < err_tol).all()


@pytest.mark.parametrize("n_rows", [64, 35, 16, 5])
@pytest.mark.parametrize("err_tol", [0.00001])
def test_nvt_tf_rossmann_inference(n_rows, err_tol):
    workflow_path = os.path.join(os.path.expanduser(MODEL_DIR), "rossmann_nvt/1/workflow")
    data_path = os.path.join(os.path.expanduser(DATA_DIR), "test_inference_rossmann_data.csv")
    actual_output_filename = os.path.join(os.path.expanduser(DATA_DIR), "rossmann_predictions.csv")

    workflow = nvt.Workflow.load(workflow_path)
    batch = cudf.read_csv(data_path, nrows=n_rows)[workflow.column_group.input_column_names]

    columns = [(col, batch[col]) for col in batch.columns]

    inputs = []
    for i, (name, col) in enumerate(columns):
        d = col.values_host.astype(col.dtype)
        d = d.reshape(len(d), 1)
        inputs.append(grpcclient.InferInput(name, d.shape, np_to_triton_dtype(col.dtype)))
        inputs[i].set_data_from_numpy(d)

    outputs = [grpcclient.InferRequestedOutput("tf.math.multiply_1")]

    with test_utils.run_triton_server(os.path.expanduser(MODEL_DIR), TRITON_SERVER_PATH) as client:
        response = client.infer("rossmann", inputs, request_id="1", outputs=outputs)

    output_actual = cudf.read_csv(os.path.expanduser(actual_output_filename), nrows=n_rows)
    output_actual = cp.asnumpy(output_actual["0"].values)
    output_predict = response.as_numpy("tf.math.multiply_1")

    diff = abs(output_actual - output_predict[:, 0])
    assert (diff < err_tol).all()
