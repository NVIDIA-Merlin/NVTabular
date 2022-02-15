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
import logging
import sys
import traceback
from typing import List

import cupy as cp
import triton_python_backend_utils as pb_utils
from triton_python_backend_utils import (
    InferenceRequest,
    InferenceResponse,
    Tensor,
    get_input_tensor_by_name,
)

from nvtabular.inference.graph.op_runner import OperatorRunner
from nvtabular.inference.graph.ops.operator import InferenceDataFrame

LOG = logging.getLogger("nvtabular")


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.runner = OperatorRunner(self.model_config)

    def execute(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Transforms the input batches by running through a NVTabular workflow.transform
        function.
        """
        params = self.model_config["parameters"]
        op_names = json.loads(params["operator_names"]["string_value"])
        first_operator_name = op_names[0]
        operator_params = json.loads(params[first_operator_name]["string_value"])
        input_column_names = list(json.loads(operator_params["input_dict"]).keys())

        responses = []

        for request in requests:
            try:
                # transform the triton tensors to a dict of name:numpy tensor
                input_tensors = {
                    name: get_input_tensor_by_name(request, name).as_numpy()
                    # name: cp.fromDlpack(get_input_tensor_by_name(request, name).to_dlpack())
                    for name in input_column_names
                }

                inf_df = InferenceDataFrame(input_tensors)

                raw_tensor_tuples = self.runner.execute(inf_df)

                # tensors = { 
                #   name:(data if hasattr(data, "get") else cp.ndarray(data)) 
                #   for name,data in raw_tensor_tuples 
                # }
                tensors = {
                    name: (data.get() if hasattr(data, "get") else data)
                    for name, data in raw_tensor_tuples
                }

                result = [Tensor(name, data) for name, data in tensors.items()]

                responses.append(InferenceResponse(result))

            except Exception as exc:  # noqa
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_string = repr(traceback.extract_tb(exc_traceback))
                # tb_string = traceback.format_exc(exc_traceback)
                responses.append(
                    pb_utils.InferenceResponse(
                        tensors=[], error=f"{exc_type}, {exc_value}, {tb_string}"
                    )
                )

        return responses
