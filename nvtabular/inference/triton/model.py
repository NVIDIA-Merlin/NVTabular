# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import os
from typing import List

import cudf
from cudf.core.column import as_column, build_column
from cudf.utils.dtypes import is_list_dtype
from triton_python_backend_utils import (
    InferenceRequest,
    InferenceResponse,
    Tensor,
    get_input_tensor_by_name,
    get_output_config_by_name,
    triton_string_to_numpy,
)

import nvtabular


class TritonPythonModel:
    """ Generic TritonPythonModel for nvtabular workflows """

    def initialize(self, args):
        workflow_path = os.path.join(
            args["model_repository"], str(args["model_version"]), "workflow"
        )
        self.workflow = nvtabular.Workflow.load(workflow_path)
        self.model_config = json.loads(args["model_config"])

        self.input_dtypes = {
            col: dtype
            for col, dtype in self.workflow.input_dtypes.items()
            if not is_list_dtype(dtype)
        }
        self.input_multihots = {
            col: dtype for col, dtype in self.workflow.input_dtypes.items() if is_list_dtype(dtype)
        }

        self.output_dtypes = dict()
        for name, dtype in self.workflow.output_dtypes.items():
            if not is_list_dtype(dtype):
                self._set_output_dtype(name)
            else:
                self._set_output_dtype(name + "__nnzs")
                self._set_output_dtype(name + "__values")

    def _set_output_dtype(self, name):
        conf = get_output_config_by_name(self.model_config, name)
        self.output_dtypes[name] = triton_string_to_numpy(conf["data_type"])

    def execute(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Transforms the input batches by running through a NVTabular workflow.transform
        function.
        """
        responses = []
        for request in requests:
            # create a cudf DataFrame from the triton request
            input_df = cudf.DataFrame(
                {
                    name: _convert_tensor(get_input_tensor_by_name(request, name))
                    for name in self.input_dtypes
                }
            )

            for name, dtype in self.input_multihots.items():
                values = as_column(
                    _convert_tensor(get_input_tensor_by_name(request, name + "__values"))
                )
                nnzs = as_column(
                    _convert_tensor(get_input_tensor_by_name(request, name + "__nnzs"))
                )
                input_df[name] = build_column(
                    None, dtype=dtype, size=nnzs.size - 1, children=(nnzs, values)
                )

            # use our NVTabular workflow to transform the dataframe
            output_df = nvtabular.workflow._transform_partition(
                input_df, [self.workflow.column_group]
            )

            # convert back to a triton response
            output_tensors = []
            for name in output_df.columns:
                col = output_df[name]
                if is_list_dtype(col.dtype):
                    # convert list values to match TF dataloader
                    values = col.list.leaves.values_host.astype(
                        self.output_dtypes[name + "__values"]
                    )
                    values = values.reshape(len(values), 1)
                    output_tensors.append(Tensor(name + "__values", values))

                    offsets = col._column.offsets.values_host.astype(
                        self.output_dtypes[name + "__nnzs"]
                    )
                    nnzs = offsets[1:] - offsets[:-1]
                    nnzs = nnzs.reshape(len(nnzs), 1)
                    output_tensors.append(Tensor(name + "__nnzs", nnzs))
                else:
                    d = col.values_host.astype(self.output_dtypes[name])
                    d = d.reshape(len(d), 1)
                    output_tensors.append(Tensor(name, d))

            responses.append(InferenceResponse(output_tensors))

        return responses


def _convert_tensor(t):
    out = t.as_numpy()
    if len(out.shape) == 2:
        out = out[:, 0]
    # cudf doesn't seem to handle dtypes like |S15
    if out.dtype.kind == "S" and out.dtype.str.startswith("|S"):
        out = out.astype("str")
    return out
