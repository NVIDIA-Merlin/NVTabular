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

from triton_python_backend_utils import InferenceRequest, InferenceResponse

from .runners.hugectr import HugeCTRWorkflowRunner
from .runners.pytorch import PyTorchWorkflowRunner
from .runners.tensorflow import TensorflowWorkflowRunner

LOG = logging.getLogger("nvtabular")

RUNNER_TYPES = {
    "hugectr": HugeCTRWorkflowRunner,
    "pytorch": PyTorchWorkflowRunner,
    "tensorflow": TensorflowWorkflowRunner,
}


class TritonPythonModel:
    def initialize(self, args):
        workflow_path = os.path.join(
            args["model_repository"], str(args["model_version"]), "workflow"
        )
        model_kind = args["model_instance_kind"]
        model_config = json.loads(args["model_config"])
        output_model = model_config["parameters"]["output_model"]["string_value"]

        if output_model == "hugectr":
            runner_class = HugeCTRWorkflowRunner
        elif output_model == "pytorch":
            runner_class = PyTorchWorkflowRunner
        elif output_model == "tensorflow":
            runner_class = TensorflowWorkflowRunner
        else:
            runner_class = TensorflowWorkflowRunner

        self.workflow = runner_class(workflow_path, model_kind, model_config)

    def execute(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        return self.workflow.execute(requests)
