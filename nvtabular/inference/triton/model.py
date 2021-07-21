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

import numpy as np
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
from nvtabular.dispatch import _concat_columns
from nvtabular.inference.triton import _convert_tensor, get_column_types
from nvtabular.inference.triton.data_conversions import convert_format
from nvtabular.ops.operator import Supports

LOG = logging.getLogger("nvtabular")


class TritonPythonModel:
    """Generic TritonPythonModel for nvtabular workflows"""

    def _initialize_ops(self, column_group, visited=None):
        if visited is None:
            visited = set()

        if column_group.op:
            inference_op = column_group.op.inference_initialize(
                column_group.columns, self.model_config
            )
            if inference_op:
                column_group.op = inference_op

        for parent in column_group.parents:
            if parent not in visited:
                visited.add(parent)
                self._initialize_ops(parent, visited)

    def initialize(self, args):
        workflow_path = os.path.join(
            args["model_repository"], str(args["model_version"]), "workflow"
        )
        self.workflow = nvtabular.Workflow.load(workflow_path)
        self.model_config = json.loads(args["model_config"])
        self.output_model = self.model_config["parameters"]["output_model"]["string_value"]

        # recurse over all column groups, initializing operators for inference pipeline
        self._initialize_ops(self.workflow.column_group)

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
                # pytorch + hugectr don't support multihot output features at inference
                if self.output_model in {"hugectr", "pytorch"}:
                    raise ValueError(f"{self.output_model} doesn't yet support multihot features")

                self._set_output_dtype(name + "__nnzs")
                self._set_output_dtype(name + "__values")

        if self.output_model == "hugectr":
            self.column_types = get_column_types(workflow_path)
            self.offsets = get_hugectr_offsets(workflow_path)
            if self.offsets is None and "cats" in self.column_types:
                raise Exception("slot_size_array.json could not be found to read the slot sizes")
        else:
            self.column_types = self.offsets = None

    def _set_output_dtype(self, name):
        conf = get_output_config_by_name(self.model_config, name)
        self.output_dtypes[name] = triton_string_to_numpy(conf["data_type"])

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

            # use our NVTabular workflow to transform the dataset
            transformed, kind = _transform_tensors(input_tensors, self.workflow.column_group)

            # if we don't have tensors in numpy format, convert back so that the we can return
            # to triton
            if kind != Supports.CPU_DICT_ARRAY:
                transformed, kind = convert_format(transformed, kind, Supports.CPU_DICT_ARRAY)

            # convert to the format expected by the DL models
            if self.output_model == "hugectr":
                response = self._transform_hugectr_outputs(transformed)
            else:
                response = self._transform_outputs(transformed)
            responses.append(response)

        return responses

    def _transform_outputs(self, tensors):
        """ transforms outputs for both pytorch and tensorflow """
        output_tensors = []
        for name, value in tensors.items():
            if isinstance(value, tuple):
                # convert list values to match TF dataloader
                values = value[0].astype(self.output_dtypes[name + "__values"])
                values = values.reshape(len(values), 1)
                output_tensors.append(Tensor(name + "__values", values))

                offsets = value[1].astype(self.output_dtypes[name + "__nnzs"])
                nnzs = offsets[1:] - offsets[:-1]
                nnzs = nnzs.reshape(len(nnzs), 1)
                output_tensors.append(Tensor(name + "__nnzs", nnzs))
            else:
                d = value.astype(self.output_dtypes[name])
                d = d.reshape(len(d), 1)
                output_tensors.append(Tensor(name, d))
        return InferenceResponse(output_tensors)

    def _transform_hugectr_outputs(self, tensors):
        output_tensors = []
        if "conts" in self.column_types:
            output_tensors.append(
                Tensor(
                    "DES",
                    _convert_to_hugectr(self.column_types["conts"], tensors, np.float32),
                )
            )
        else:
            output_tensors.append(Tensor("DES", np.array([[]], np.float32)))

        if "cats" in self.column_types:
            cats_np = _convert_to_hugectr(self.column_types["cats"], tensors, np.int64)
            cats_np += self.offsets
            output_tensors.append(
                Tensor(
                    "CATCOLUMN",
                    cats_np,
                )
            )
        else:
            output_tensors.append(Tensor("CATCOLUMN", np.array([[]], np.int64)))

        len_cats_np = cats_np.shape[1]
        row_index = np.arange(len_cats_np + 1, dtype=np.int32).reshape(1, len_cats_np + 1)
        output_tensors.append(Tensor("ROWINDEX", row_index))

        return InferenceResponse(output_tensors)


def _convert_to_hugectr(columns, tensors, dtype):
    """ converts outputs to a numpy input compatible with hugectr """
    rows = max(len(tensors[name]) for name in columns)
    d = np.empty((rows, len(columns)), dtype=dtype)
    for i, name in enumerate(columns):
        d[:, i] = tensors[name].astype(dtype)
    return d.reshape(1, len(columns) * rows)


def get_hugectr_offsets(path):
    if os.path.exists(path):
        slot_sizes = json.load(open(os.path.join(path, "slot_size_array.json")))
        slot_sizes = slot_sizes["slot_size_array"]
        slot_sizes.insert(0, 0)
        slot_sizes.pop()
        return np.cumsum(np.array(slot_sizes)).tolist()
    else:
        return None


def _transform_tensors(input_tensors, column_group):
    if column_group.parents:
        tensors, kind = None, None
        for parent in column_group.parents:
            transformed, transformed_kind = _transform_tensors(input_tensors, parent)
            if tensors is None:
                tensors, kind = transformed, transformed_kind
            else:
                if kind != transformed_kind:
                    # we have multiple different kinds of data here (dataframe/array on cpu/gpu)
                    # we need to convert to a common format here first before concatentating.
                    op = column_group.op
                    target_kind = op.supports if op else Supports.CPU_DICT_ARRAY
                    # note : the 2nd convert_format call needs to be stricter in what the kind is
                    # (exact match rather than a bitmask of values)
                    tensors, kind = convert_format(tensors, kind, target_kind)
                    transformed, _ = convert_format(transformed, transformed_kind, kind)

                tensors = _concat_tensors([tensors, transformed], kind)

    else:
        tensors = {c: input_tensors[c] for c in column_group.columns}
        kind = Supports.CPU_DICT_ARRAY

    if column_group.op:
        try:
            # if the op doesn't support the current kind - we need to convert
            if not column_group.op.supports & kind:
                tensors, kind = convert_format(tensors, kind, column_group.op.supports)

            tensors = column_group.op.transform(column_group.columns, tensors)

        except Exception:
            LOG.exception("Failed to transform operator %s", column_group.op)
            raise

    return tensors, kind


def _concat_tensors(tensors, kind):
    if kind & (Supports.GPU_DATAFRAME | Supports.CPU_DATAFRAME):
        return _concat_columns(tensors)
    else:
        output = tensors[0]
        for tensor in tensors[1:]:
            output.update(tensor)
        return output
