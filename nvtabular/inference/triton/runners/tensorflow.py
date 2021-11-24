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
from pathlib import Path

from triton_python_backend_utils import (
    InferenceResponse,
    Tensor,
    get_output_config_by_name,
    triton_string_to_numpy,
)

from nvtabular.dispatch import _is_list_dtype
from nvtabular.inference.triton.runners.base import WorkflowRunner


class TensorflowWorkflowRunner(WorkflowRunner):
    def __init__(self, workflow_path, model_kind, model_config):
        super().__init__(workflow_path, model_kind, model_config)

        self.column_types = self.offsets = None
        self.output_dtypes = dict()
        for name, dtype in self.workflow.output_dtypes.items():
            if not _is_list_dtype(dtype):
                self._set_output_dtype(name)
            else:
                self._set_output_dtype(name + "__nnzs")
                self._set_output_dtype(name + "__values")

    def _set_output_dtype(self, name):
        conf = get_output_config_by_name(self.model_config, name)
        self.output_dtypes[name] = triton_string_to_numpy(conf["data_type"])

    def _transform_outputs(self, tensors):
        # Load extra info needed for the Transformer4Rec (if exists)
        model_info_path = os.path.join(self.workflow_path, "model_info.json")
        model_info_file = Path(model_info_path)
        sparse_feat = {}
        if model_info_file.exists():
            with open(model_info_path) as json_file:
                sparse_feat = json.load(json_file)

        # transforms outputs for both pytorch and tensorflow
        output_tensors = []
        for name, value in tensors.items():
            if isinstance(value, tuple) and name not in sparse_feat.keys():
                # convert list values to match TF dataloader
                values = value[0].astype(self.output_dtypes[name + "__values"])
                values = values.reshape(len(values), 1)
                output_tensors.append(Tensor(name + "__values", values))
                offsets = value[1].astype(self.output_dtypes[name + "__nnzs"])
                nnzs = offsets[1:] - offsets[:-1]
                nnzs = nnzs.reshape(len(nnzs), 1)
                output_tensors.append(Tensor(name + "__nnzs", nnzs))
            elif isinstance(value, tuple) and name in sparse_feat.keys():
                # convert sparse tensors to dense representations
                d = value[0].astype(self.output_dtypes[name])
                col_dim = sparse_feat[name]
                row_dim = d.shape[0] // col_dim
                d = d.reshape(row_dim, col_dim)
                output_tensors.append(Tensor(name, d))
            else:
                d = value.astype(self.output_dtypes[name])
                d = d.reshape(len(d), 1)
                output_tensors.append(Tensor(name, d))
        return InferenceResponse(output_tensors)
