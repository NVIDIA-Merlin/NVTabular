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
import os
from typing import List

from triton_python_backend_utils import (
    InferenceRequest,
    InferenceResponse,
    Tensor,
    get_input_tensor_by_name,
)

import nvtabular
from nvtabular.dispatch import _is_list_dtype
from nvtabular.inference.triton import _convert_tensor, get_column_types
from nvtabular.inference.triton.workflow.hugectr import HugeCTRWorkflowRunner
from nvtabular.inference.triton.workflow.pytorch import PyTorchWorkflowRunner
from nvtabular.inference.triton.workflow.tensorflow import TensorflowWorkflowRunner

LOG = logging.getLogger("nvtabular")

RUNNER_TYPES = {
    "hugectr": HugeCTRWorkflowRunner,
    "pytorch": PyTorchWorkflowRunner,
    "tensorflow": TensorflowWorkflowRunner,
}


class TritonPythonModel:
    def initialize(self, args):
        # Arg parsing
        workflow_path = os.path.join(
            args["model_repository"], str(args["model_version"]), "workflow"
        )
        model_device = args["model_instance_kind"]

        # Workflow instantiation
        self.workflow = nvtabular.Workflow.load(workflow_path)

        # Config loading and parsing
        model_config = json.loads(args["model_config"])
        model_framework = model_config["parameters"]["output_model"]["string_value"]

        column_types = get_column_types(workflow_path)
        self.runner = _construct_workflow_runner(
            self.workflow, column_types, model_framework, model_config, model_device
        )

        # Dtype parsing
        input_dtypes = self.workflow.input_dtypes.items()
        self.input_dtypes, self.input_multihots = _parse_input_dtypes(input_dtypes)

    def execute(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Transforms the input batches by running through a NVTabular workflow.transform
        function.
        """
        responses = []
        for request in requests:
            # transform the triton tensors to a dict of name:numpy tensor
            input_tensors = {
                name: _convert_tensor(get_input_tensor_by_name(request, name))
                for name in self.input_dtypes
            }

            # multihots are represented as a tuple of (values, offsets)
            for name, dtype in self.input_multihots.items():
                values = _convert_tensor(get_input_tensor_by_name(request, name + "__values"))
                offsets = _convert_tensor(get_input_tensor_by_name(request, name + "__nnzs"))
                input_tensors[name] = (values, offsets)

            raw_tensor_tuples = self.runner.run_workflow(input_tensors)

            result = [Tensor(name, data) for name, data in raw_tensor_tuples]

            responses.append(InferenceResponse(result))

        return responses


def _construct_workflow_runner(workflow, column_types, model_framework, model_config, model_device):
    if model_framework == "hugectr":
        runner_class = HugeCTRWorkflowRunner
    elif model_framework == "pytorch":
        runner_class = PyTorchWorkflowRunner
    else:
        runner_class = TensorflowWorkflowRunner

    return runner_class(workflow, column_types, model_config, model_device)


def _parse_input_dtypes(dtypes):
    input_dtypes = {col: dtype for col, dtype in dtypes if not _is_list_dtype(dtype)}
    input_multihots = {col: dtype for col, dtype in dtypes if _is_list_dtype(dtype)}

    return input_dtypes, input_multihots
