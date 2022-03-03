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

import numpy as np

from nvtabular.inference.workflow.base import WorkflowRunner
from nvtabular.ops import get_embedding_sizes


class HugeCTRWorkflowRunner(WorkflowRunner):
    def __init__(self, workflow, output_dtypes, model_config, model_device):
        super().__init__(workflow, output_dtypes, model_config, model_device)

        if self.cats:
            self.offsets = self.get_offsets(self.workflow, self.cats)

    def _transform_outputs(self, tensors):
        output_tensors = []
        if self.conts:
            output_tensors.append(
                (
                    "DES",
                    self._convert(self.conts, tensors, np.float32),
                )
            )
        else:
            output_tensors.append(("DES", np.array([[]], np.float32)))

        if self.cats:
            for name in self.cats:
                tensors[name] += self.offsets[name]
            cats_np = self._convert(self.cats, tensors, np.int64)
            output_tensors.append(
                (
                    "CATCOLUMN",
                    cats_np,
                )
            )
        else:
            output_tensors.append(("CATCOLUMN", np.array([[]], np.int64)))

        len_cats_np = cats_np.shape[1]
        row_index = np.arange(len_cats_np + 1, dtype=np.int32).reshape(1, len_cats_np + 1)
        output_tensors.append(("ROWINDEX", row_index))

        return output_tensors

    def _convert(self, columns, tensors, dtype):
        """converts outputs to a numpy input compatible with hugectr"""
        rows = max(len(tensors[name]) for name in columns)
        d = self._convert_to_np(columns, tensors, dtype, rows)
        return d.reshape(1, len(columns) * rows)

    def get_offsets(self, workflow, categorical_cols):
        embeddings = get_embedding_sizes(workflow)
        if embeddings is None:
            raise Exception("embeddings cannot be None")
        else:
            offsets = dict()
            curr_offset = 0
            for name in categorical_cols:
                offsets[name] = curr_offset
                curr_offset += embeddings[name][0]
            return offsets
