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

import datetime as dt
import os

import cupy as cp
import pandas as pd

import nvtabular as nvt


def _run_query(
    client,
    n_rows,
    model_name,
    workflow_path,
    data_path,
    actual_output_filename,
    output_name,
    input_cols_name=None,
    backend="tensorflow",
):

    import tritonclient.grpc as grpcclient
    from tritonclient.utils import np_to_triton_dtype

    workflow = nvt.Workflow.load(workflow_path)

    if input_cols_name is None:
        batch = pd.read_csv(data_path, nrows=n_rows)[workflow.output_node.input_columns.names]
    else:
        batch = pd.read_csv(data_path, nrows=n_rows)[input_cols_name]

    input_dtypes = workflow.input_dtypes
    columns = [(col, batch[col]) for col in batch.columns]

    inputs = []
    for i, (name, col) in enumerate(columns):
        d = col.fillna(0).values.astype(input_dtypes[name])
        d = d.reshape(len(d), 1)
        inputs.append(grpcclient.InferInput(name, d.shape, np_to_triton_dtype(input_dtypes[name])))
        inputs[i].set_data_from_numpy(d)

    outputs = [grpcclient.InferRequestedOutput(output_name)]
    time_start = dt.datetime.now()
    response = client.infer(model_name, inputs, request_id="1", outputs=outputs)
    run_time = dt.datetime.now() - time_start

    output_key = "output" if backend == "hugectr" else "0"

    output_actual = pd.read_csv(os.path.expanduser(actual_output_filename), nrows=n_rows)
    output_actual = cp.asnumpy(output_actual[output_key].values)
    output_predict = response.as_numpy(output_name)

    if backend == "tensorflow":
        output_predict = output_predict[:, 0]

    diff = abs(output_actual - output_predict)
    return diff, run_time
