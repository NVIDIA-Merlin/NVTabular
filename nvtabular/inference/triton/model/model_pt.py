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

import cloudpickle
import torch
import triton_python_backend_utils as pb_utils

from nvtabular.inference.triton import _convert_string2pytorch_dtype, _convert_tensor

LOG = logging.getLogger("nvtabular")

sparse_value_marker = "__values"
sparse_nnzs_marker = "__nnzs"


class TritonPythonModel:
    """Generic TritonPythonModel for nvtabular workflows"""

    def initialize(self, args):
        # Arg parsing
        repository_path = pathlib.Path(args["model_repository"])
        model_version = str(args["model_version"])

        # Handle bug in Tritonserver 22.06
        # model_repository argument became path to model.py
        if str(repository_path).endswith(".py"):
            repository_path = repository_path.parent.parent

        model_path = repository_path / model_version / "model.pkl"

        # Load the pickled PyTorch model
        self.model = cloudpickle.load(
            open(str(model_path), "rb")  # pylint: disable=consider-using-with
        )

        # Load the state dict of the PyTorch model
        model_path = repository_path / model_version / "model.pth"
        self.model.load_state_dict(torch.load(str(model_path)))
        self.model.eval()

        # Load model config file
        self.model_config = json.loads(args["model_config"])

        # Load extra info needed for the Transformer4Rec (if exists)
        model_info_path = repository_path / model_version / "model_info.json"
        self.model_info = None
        model_info_file = pathlib.Path(model_info_path)
        if model_info_file.exists():
            with open(str(model_info_path), encoding="utf-8") as json_file:
                self.model_info = json.load(json_file)

        # Get the name of the dense and sparse inputs, and the outputs
        self.inputs = {}
        self.sparse_inputs = {}
        self.outputs = {}
        len_svm = len(sparse_value_marker)
        len_snm = len(sparse_nnzs_marker)

        for val in self.model_config["input"]:
            name = val["name"]

            # NVTabular adds this specific marker "__values" into the name of the sparse inputs
            # The ones that has the marker "__nnzs" are for the sparse values
            # Hence, dense and sparse inputs are identified based on these markers
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

    def execute(self, requests):
        """Predicts the input batches by running through a PyTorch predict function."""

        # To be able to execute the queries, the PyTorch model must accept a dict input
        # and generates a dict output that has the output in the the "predictions"
        # bucket. Otherwise, it'll throw an error.

        with torch.no_grad():
            responses = []
            for request in requests:
                # Convert the input data to dict to pass it into the PyTorch model
                input_dict = {}
                for name, dtype in self.inputs.items():
                    input_dict[name] = torch.tensor(
                        _convert_tensor(pb_utils.get_input_tensor_by_name(request, name)),
                        dtype=dtype,
                    ).cuda()

                # Sparse inputs have a special format
                for name, dtype in self.sparse_inputs.items():
                    # Convert to fixed dtypes if requested
                    if self.model_info["use_fix_dtypes"]:
                        dtype = _convert_dtype(dtype)

                    # Get __values and __nnzs
                    input_val = _convert_tensor(
                        pb_utils.get_input_tensor_by_name(request, name + sparse_value_marker)
                    )
                    input_nnzs = _convert_tensor(
                        pb_utils.get_input_tensor_by_name(request, name + sparse_nnzs_marker)
                    )
                    input_nnzs = torch.tensor(input_nnzs, dtype=torch.int64)
                    input_values = torch.tensor(input_val, dtype=dtype)

                    # Get the PyTorch sparse_coo_tensor
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

                # Call forward function to get the predictions
                # Forward function should return a dict with the "predictions" bucket
                out = self.model(input_dict, training=False)
                if not isinstance(out, dict):
                    raise ValueError("output of the forward function should be a dict")

                # Get the predictions from the out
                pred = out.get("predictions")
                if pred is None:
                    raise KeyError(
                        "output of the forward function should have a bucket named as predictions"
                    )

                # There is one output in the config file
                # since the PyTorch models generate a tensor as an output
                output_info = self.model_config["output"][0]
                output_tensor = pb_utils.Tensor(output_info["name"], pred.cpu().detach().numpy())
                responses.append(pb_utils.InferenceResponse([output_tensor]))

        return responses


def _get_indices(nnzs, device="cuda"):
    """Calculate indices for the PyTorch sparse_coo_tensor"""
    offsets = torch.cat((torch.tensor([1]), nnzs), 0)
    offsets = offsets.cumsum(0)
    row_ids = torch.arange(len(offsets) - 1)
    row_ids_repeated = torch.repeat_interleave(row_ids, nnzs)
    row_offset_repeated = torch.repeat_interleave(offsets[:-1], nnzs)
    col_ids = torch.arange(len(row_offset_repeated)) - row_offset_repeated + 1
    indices = torch.cat([row_ids_repeated.unsqueeze(-1), col_ids.unsqueeze(-1)], axis=1)
    return indices.T


def _get_sparse_tensor(values, indices, num_rows, seq_limit, sparse_as_dense, device="cuda"):
    """Creates the PyTorch sparse_coo_tensor"""
    sparse_tensor = torch.sparse_coo_tensor(
        indices, values, torch.Size([num_rows, seq_limit]), device=device
    )
    if sparse_as_dense:
        sparse_tensor = sparse_tensor.to_dense()
    return sparse_tensor


def _build_sparse_tensor(values, nnzs, seq_limit, sparse_as_dense, device="cuda"):
    """Builds PyTorch sparse_coo_tensor by converting the __values and __nnzs inputs"""
    indices = _get_indices(nnzs, device)
    num_rows = len(nnzs)
    return _get_sparse_tensor(values, indices, num_rows, seq_limit, sparse_as_dense, device)


def _convert_dtype(dtype):
    """Transformer4Rec uses these fixed dtypes and this function converts the original dtype
    to this fixed dtypes"""
    if dtype in [torch.float64, torch.float32, torch.float16]:
        return torch.float32
    if dtype in [
        torch.int64,
        torch.int32,
        torch.int16,
        torch.int8,
        torch.uint8,
    ]:
        return torch.long

    raise ValueError(f"Can't convert dtype {dtype})")
