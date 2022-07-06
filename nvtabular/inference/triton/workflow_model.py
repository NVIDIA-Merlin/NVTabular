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
import pathlib

import numpy as np
import triton_python_backend_utils as pb_utils

import nvtabular
from merlin.core.dispatch import is_list_dtype
from nvtabular.inference.triton import _convert_tensor
from nvtabular.inference.workflow.hugectr import HugeCTRWorkflowRunner
from nvtabular.inference.workflow.pytorch import PyTorchWorkflowRunner
from nvtabular.inference.workflow.tensorflow import TensorflowWorkflowRunner

LOG = logging.getLogger("nvtabular")


class TritonPythonModel:
    """
    Triton model used for running NVT Workflows
    """

    def initialize(self, args):
        # Arg parsing
        repository_path = pathlib.Path(args["model_repository"])

        # Handle bug in Tritonserver 22.06
        # model_repository argument became path to model.py
        if str(repository_path).endswith(".py"):
            repository_path = repository_path.parent.parent

        workflow_path = repository_path / str(args["model_version"]) / "workflow"
        model_device = args["model_instance_kind"]

        # Workflow instantiation
        self.workflow = nvtabular.Workflow.load(str(workflow_path))

        # Config loading and parsing
        self.model_config = json.loads(args["model_config"])
        model_framework = self.model_config["parameters"]["output_model"]["string_value"]

        # Dtype parsing
        input_dtypes = self.workflow.input_dtypes.items()
        self.input_dtypes, self.input_multihots = _parse_input_dtypes(input_dtypes)

        self.output_dtypes = {}
        if model_framework == "hugectr":
            self.output_dtypes = {"DES": np.float32, "CATCOLUMN": np.int64, "ROWINDEX": np.int32}
        else:
            for col_name, col_schema in self.workflow.output_schema.column_schemas.items():
                if col_schema.is_list and col_schema.is_ragged:
                    self._set_output_dtype(col_name + "__nnzs")
                    self._set_output_dtype(col_name + "__values")
                else:
                    self._set_output_dtype(col_name)

        if model_framework == "hugectr":
            runner_class = HugeCTRWorkflowRunner
        elif model_framework == "pytorch":
            runner_class = PyTorchWorkflowRunner
        else:
            runner_class = TensorflowWorkflowRunner

        self.runner = runner_class(
            self.workflow, self.output_dtypes, self.model_config, model_device
        )

    def _set_output_dtype(self, name):
        conf = pb_utils.get_output_config_by_name(self.model_config, name)
        self.output_dtypes[name] = pb_utils.triton_string_to_numpy(conf["data_type"])

    def execute(self, requests):
        """Transforms the input batches by running through a NVTabular workflow.transform
        function.
        """
        responses = []
        for request in requests:
            # transform the triton tensors to a dict of name:numpy tensor
            input_tensors = {
                name: _convert_tensor(pb_utils.get_input_tensor_by_name(request, name))
                for name in self.input_dtypes
            }

            # multihots are represented as a tuple of (values, offsets)
            for name, dtype in self.input_multihots.items():
                values = _convert_tensor(
                    pb_utils.get_input_tensor_by_name(request, name + "__values")
                )
                offsets = _convert_tensor(
                    pb_utils.get_input_tensor_by_name(request, name + "__nnzs")
                )
                input_tensors[name] = (values, offsets)

            raw_tensor_tuples = self.runner.run_workflow(input_tensors)

            result = [pb_utils.Tensor(name, data) for name, data in raw_tensor_tuples]

            responses.append(pb_utils.InferenceResponse(result))

        return responses


def _parse_input_dtypes(dtypes):
    input_dtypes = {col: dtype for col, dtype in dtypes if not is_list_dtype(dtype)}
    input_multihots = {col: dtype for col, dtype in dtypes if is_list_dtype(dtype)}

    return input_dtypes, input_multihots
