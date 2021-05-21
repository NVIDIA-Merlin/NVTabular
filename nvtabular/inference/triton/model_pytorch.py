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
import numpy as np
from triton_python_backend_utils import (
    InferenceRequest,
    InferenceResponse,
    Tensor,
    get_input_tensor_by_name,
)

import nvtabular
from nvtabular.inference.triton import _convert_tensor, get_column_types


class TritonPythonModel:
    """Generic TritonPythonModel for nvtabular workflows"""

    def initialize(self, args):
        workflow_path = os.path.join(
            args["model_repository"], str(args["model_version"]), "workflow"
        )
        self.workflow = nvtabular.Workflow.load(workflow_path)
        self.model_config = json.loads(args["model_config"])
        self.output_columns = get_column_types(workflow_path)

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
                    for name in self.workflow.column_group.input_column_names
                }
            )

            # use our NVTabular workflow to transform the dataframe
            output_df = nvtabular.workflow._transform_partition(
                input_df, [self.workflow.column_group]
            )

            output_tensors = []
            for col, val in self.output_columns.items():
                d = _convert_cudf2numpy(output_df[val["columns"]], val["dtype"])
                output_tensors.append(Tensor(col, d))

            responses.append(InferenceResponse(output_tensors))

        return responses


def _convert_cudf2numpy(df, dtype):
    d = np.empty(df.shape).astype(dtype)
    for i, name in enumerate(df.columns):
        d[:, i] = df[name].values_host.astype(dtype)

    return d
