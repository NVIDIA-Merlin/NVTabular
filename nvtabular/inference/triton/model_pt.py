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

import functools
import json
import logging
import os
from typing import List
import torch
import cloudpickle
import numpy as np
from pathlib import Path

from triton_python_backend_utils import (
    InferenceRequest,
    InferenceResponse,
    Tensor,
    get_input_tensor_by_name,
    get_output_config_by_name,
    triton_string_to_numpy,
)

import nvtabular
from nvtabular.dispatch import _concat_columns, _is_list_dtype
from nvtabular.inference.triton import _convert_tensor, get_column_types, _convert_string2pytorch_dtype
from nvtabular.inference.triton.data_conversions import convert_format
from nvtabular.ops import get_embedding_sizes
from nvtabular.ops.operator import Supports

LOG = logging.getLogger("nvtabular")


class TritonPythonModel:
    """Generic TritonPythonModel for nvtabular workflows"""

    def initialize(self, args):
        model_path = os.path.join(
            args["model_repository"], str(args["model_version"]), "model.pkl"
        )
        self.model = cloudpickle.load(open(model_path, "rb"))
        model_path = os.path.join(
            args["model_repository"], str(args["model_version"]), "model.pth"
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model_config = json.loads(args["model_config"])
        
        model_info_path = os.path.join(
            args["model_repository"], str(args["model_version"]), "model_info.json"
        )
        self.model_info = None
        model_info_file = Path(model_info_path)
        if model_info_file.exists():
            with open(model_info_path) as json_file:
                self.model_info = json.load(json_file)

        self.inputs = dict()
        self.sparse_inputs = dict()
        self.outputs = dict()
        for val in self.model_config["input"]:
            name = val["name"]
            if len(name) > 8:
                if name[-8:] == "__values":
                    self.sparse_inputs[name[0:(len(name)-8)]] = _convert_string2pytorch_dtype(val["data_type"])
                elif name[-6:] != "__nnzs":
                    self.inputs[name] = _convert_string2pytorch_dtype(val["data_type"])
            else:
                if len(name) > 6:
                    if name[-6:] != "__nnzs":
                        self.inputs[name] = _convert_string2pytorch_dtype(val["data_type"])
                else:
                    self.inputs[name] = _convert_string2pytorch_dtype(val["data_type"])

        for val in self.model_config["output"]:
            self.outputs[val["name"]] = _convert_string2pytorch_dtype(val["data_type"])

    def execute(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Predicts the input batches by running through a PyTorch predict function."""
        responses = []
        for request in requests:
            input_dict = dict()
            for name, dtype in self.inputs.items():
                input_dict[name] = torch.tensor(_convert_tensor(get_input_tensor_by_name(request, name)), device='cuda:0', dtype=dtype)
            
            for name, dtype in self.sparse_inputs.items():
                input_val = _convert_tensor(get_input_tensor_by_name(request, name + "__values"))
                if self.model_info != None:
                    if self.model_info["sparse_max"].get(name) != None:
                        new_val = np.zeros((len(input_val), self.model_info["sparse_max"][name]))
                        new_val[:,0] = input_val
                        input_dict[name] = torch.tensor(new_val, device='cuda:0', dtype=dtype)
                    else:
                        input_dict[name] = torch.tensor(input_val, device='cuda:0', dtype=dtype)
                else:
                    input_dict[name] = torch.tensor(input_val, device='cuda:0', dtype=dtype)
                
            out = self.model(input_dict)
            
            output_info = self.model_config["output"][0]
            output_tensor = Tensor(output_info["name"], out.cpu().detach().numpy())
            responses.append(InferenceResponse([output_tensor]))
            
        return responses
