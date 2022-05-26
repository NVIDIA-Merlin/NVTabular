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
import json
import os

import pandas as pd
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

from merlin.core.dispatch import is_list_dtype, is_string_dtype, make_df
from nvtabular.inference.triton.ensemble import (  # noqa
    _convert_string2pytorch_dtype,
    export_hugectr_ensemble,
    export_pytorch_ensemble,
    export_tensorflow_ensemble,
    generate_hugectr_model,
    generate_nvtabular_model,
)


def convert_df_to_triton_input(column_names, batch, input_class=grpcclient.InferInput):
    columns = [(col, batch[col]) for col in column_names]
    inputs = []
    for i, (name, col) in enumerate(columns):
        if is_list_dtype(col):
            if isinstance(col, pd.Series):
                raise ValueError("this function doesn't support CPU list values yet")
            inputs.append(
                _convert_column_to_triton_input(
                    col._column.offsets.values_host.astype("int64"), name + "__nnzs", input_class
                )
            )
            inputs.append(
                _convert_column_to_triton_input(
                    col.list.leaves.values_host.astype("int64"), name + "__values", input_class
                )
            )
        else:
            values = col.values if isinstance(col, pd.Series) else col.values_host
            inputs.append(_convert_column_to_triton_input(values, name, input_class))
    return inputs


def _convert_column_to_triton_input(col, name, input_class=grpcclient.InferInput):
    col = col.reshape(len(col), 1)
    input_tensor = input_class(name, col.shape, np_to_triton_dtype(col.dtype))
    input_tensor.set_data_from_numpy(col)
    return input_tensor


def convert_triton_output_to_df(columns, response):
    return make_df({col: response.as_numpy(col) for col in columns})


def get_column_types(path):
    return json.load(open(os.path.join(path, "column_types.json")))


def _convert_tensor(t):
    out = t.as_numpy()
    if len(out.shape) == 2:
        out = out[:, 0]
    # cudf doesn't seem to handle dtypes like |S15 or object that well
    if is_string_dtype(out.dtype):
        out = out.astype("str")
    return out
