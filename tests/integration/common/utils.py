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
import itertools
import json
import os
import shutil
import subprocess
import sys

import cudf
import cupy as cp
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

import nvtabular as nvt


def _run_notebook(
    tmpdir,
    notebook_path,
    input_path,
    output_path,
    batch_size=None,
    gpu_id=0,
    clean_up=True,
    transform=None,
    params=None,
    main_block=-1,
):
    params = params or []

    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU_TARGET_ID", gpu_id)

    if not os.path.exists(input_path):
        os.makedirs(input_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if batch_size:
        os.environ["BATCH_SIZE"] = os.environ.get("BATCH_SIZE", batch_size)

    os.environ["INPUT_DATA_DIR"] = input_path
    os.environ["OUTPUT_DATA_DIR"] = output_path
    # read in the notebook as JSON, and extract a python script from it
    notebook = json.load(open(notebook_path, encoding="utf-8"))
    source_cells = [cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"]

    lines = [
        transform(line.rstrip()) if transform else line
        for line in itertools.chain(*source_cells)
        if not (line.startswith("%") or line.startswith("!"))
    ]

    # Replace config params
    if params:

        def transform_fracs(line):
            line = line.replace("device_limit_frac = 0.7", "device_limit_frac = " + str(params[0]))
            line = line.replace("device_pool_frac = 0.8", "device_pool_frac = " + str(params[1]))
            return line.replace("part_mem_frac = 0.15", "part_mem_frac = " + str(params[2]))

        lines = [transform_fracs(line) for line in lines]

    # Add guarding block and indentation
    if main_block >= 0:
        lines.insert(main_block, 'if __name__ == "__main__":')
        for i in range(main_block + 1, len(lines)):
            lines[i] = "    " + lines[i]

    # save the script to a file, and run with the current python executable
    # we're doing this in a subprocess to avoid some issues using 'exec'
    # that were causing a segfault with globals of the exec'ed function going
    # out of scope
    script_path = os.path.join(tmpdir, "notebook.py")
    with open(script_path, "w") as script:
        script.write("\n".join(lines))
    output = subprocess.check_output([sys.executable, script_path])
    # save location will default to run location
    output = output.decode("utf-8")
    _, note_name = os.path.split(notebook_path)
    note_name = note_name.split(".")[0]
    if output:
        with open(f"test_res_{note_name}", "w+") as w_file:
            w_file.write(output)
    # clear out products
    if clean_up:
        shutil.rmtree(output_path)
    return output


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

    workflow = nvt.Workflow.load(workflow_path)

    if input_cols_name is None:
        batch = cudf.read_csv(data_path, nrows=n_rows)[workflow.output_node.input_columns.names]
    else:
        batch = cudf.read_csv(data_path, nrows=n_rows)[input_cols_name]

    input_dtypes = workflow.input_dtypes
    columns = [(col, batch[col]) for col in batch.columns]

    inputs = []
    for i, (name, col) in enumerate(columns):
        d = col.values_host.astype(input_dtypes[name])
        d = d.reshape(len(d), 1)
        inputs.append(grpcclient.InferInput(name, d.shape, np_to_triton_dtype(input_dtypes[name])))
        inputs[i].set_data_from_numpy(d)

    outputs = [grpcclient.InferRequestedOutput(output_name)]
    time_start = dt.datetime.now()
    response = client.infer(model_name, inputs, request_id="1", outputs=outputs)
    run_time = dt.datetime.now() - time_start

    output_key = "output" if backend == "hugectr" else "0"

    output_actual = cudf.read_csv(os.path.expanduser(actual_output_filename), nrows=n_rows)
    output_actual = cp.asnumpy(output_actual[output_key].values)
    output_predict = response.as_numpy(output_name)

    if backend == "tensorflow":
        output_predict = output_predict[:, 0]

    diff = abs(output_actual - output_predict)
    return diff, run_time
