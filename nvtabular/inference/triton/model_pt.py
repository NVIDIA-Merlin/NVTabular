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
from pathlib import Path
from typing import List

import cloudpickle
import torch
from triton_python_backend_utils import (
    InferenceRequest,
    InferenceResponse,
    Tensor,
    get_input_tensor_by_name,
)

from nvtabular.inference.triton import (
    _convert_string2pytorch_dtype,
    _convert_tensor,
)

LOG = logging.getLogger("nvtabular")

sparse_value_marker = "__values"
sparse_nnzs_marker = "__nnzs"


class TritonPythonModel:
    """Generic TritonPythonModel for nvtabular workflows"""

    def initialize(self, args):
        model_path = os.path.join(args["model_repository"], str(args["model_version"]), "model.pkl")
        self.model = cloudpickle.load(open(model_path, "rb"))
        model_path = os.path.join(args["model_repository"], str(args["model_version"]), "model.pth")
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
        len_svm = len(sparse_value_marker)
        len_snm = len(sparse_nnzs_marker)

        for val in self.model_config["input"]:
            name = val["name"]
            if len(name) > len_svm:
                if name[-len_svm:] == sparse_value_marker:
                    self.sparse_inputs[
                        name[0 : (len(name) - len_svm)]
                    ] = _convert_string2pytorch_dtype(val["data_type"])
                elif name[-len_snm:] != sparse_nnzs_marker:
                    self.inputs[name] = _convert_string2pytorch_dtype(val["data_type"])
            else:
                if len(name) > len_snm:
                    if name[-len_snm:] != sparse_nnzs_marker:
                        self.inputs[name] = _convert_string2pytorch_dtype(val["data_type"])
                else:
                    self.inputs[name] = _convert_string2pytorch_dtype(val["data_type"])

        for val in self.model_config["output"]:
            self.outputs[val["name"]] = _convert_string2pytorch_dtype(val["data_type"])

    # TODO: Use model.to('cpu') and model.to('cuda') to get cpu and gpu copies of the same model
    #       and run inference based on the batch size.
    def execute(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Predicts the input batches by running through a PyTorch predict function."""
        responses = []
        for request in requests:
            input_dict = dict()
            for name, dtype in self.inputs.items():
                input_dict[name] = torch.tensor(
                    _convert_tensor(get_input_tensor_by_name(request, name)), dtype=dtype
                ).cuda()

            for name, dtype in self.sparse_inputs.items():
                input_val = _convert_tensor(
                    get_input_tensor_by_name(request, name + sparse_value_marker)
                )
                input_nnzs = _convert_tensor(
                    get_input_tensor_by_name(request, name + sparse_nnzs_marker)
                )
                input_nnzs = torch.tensor(input_nnzs, dtype=torch.int64)
                input_values = torch.tensor(input_val, dtype=dtype)

                sparse_to_dense = False
                seq_limit = 0
                if self.model_info is not None:
                    if self.model_info["sparse_max"].get(name) is not None:
                        sparse_to_dense = True
                        seq_limit = self.model_info["sparse_max"][name]

                if seq_limit == 0:
                    seq_limit = int(input_nnzs.max())

                input_dict[name] = _build_sparse_tensor(
                    input_values, input_nnzs, seq_limit, sparse_to_dense
                )

            out = self.model(input_dict)

            output_info = self.model_config["output"][0]
            output_tensor = Tensor(output_info["name"], out["predictions"].cpu().detach().numpy())
            responses.append(InferenceResponse([output_tensor]))

        return responses


def _get_indices(nnzs, device="cuda"):
    offsets = torch.cat((torch.tensor([1]), nnzs), 0)
    offsets = offsets.cumsum(0)
    row_ids = torch.arange(len(offsets) - 1)
    row_ids_repeated = torch.repeat_interleave(row_ids, nnzs)
    row_offset_repeated = torch.repeat_interleave(offsets[:-1], nnzs)
    col_ids = torch.arange(len(row_offset_repeated)) - row_offset_repeated + 1
    indices = torch.cat([row_ids_repeated.unsqueeze(-1), col_ids.unsqueeze(-1)], axis=1)
    return indices.T


def _get_sparse_tensor(values, indices, num_rows, seq_limit, sparse_as_dense, device="cuda"):
    sparse_tensor = torch.sparse_coo_tensor(
        indices, values, torch.Size([num_rows, seq_limit]), device=device
    )
    if sparse_as_dense:
        sparse_tensor = sparse_tensor.to_dense()
    return sparse_tensor


def _build_sparse_tensor(values, nnzs, seq_limit, sparse_as_dense, device="cuda"):
    indices = _get_indices(nnzs, device)
    num_rows = len(nnzs)
    return _get_sparse_tensor(values, indices, num_rows, seq_limit, sparse_as_dense, device)
