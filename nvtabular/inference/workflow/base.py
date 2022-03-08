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
from abc import ABC, abstractmethod

import numpy as np

from merlin.core.dispatch import concat_columns
from merlin.dag import ColumnSelector, Supports
from merlin.schema import Tags
from nvtabular.inference.triton.data_conversions import convert_format

LOG = logging.getLogger("nvtabular")


class WorkflowRunner(ABC):
    def __init__(self, workflow, output_dtypes, model_config, model_device):
        self.workflow = workflow
        self.output_dtypes = output_dtypes
        self.model_config = model_config
        self.device = model_device

        output_schema = self.workflow.output_schema

        schema_cats = output_schema.apply(ColumnSelector(tags=[Tags.CATEGORICAL])).column_names
        schema_conts = output_schema.apply(ColumnSelector(tags=[Tags.CONTINUOUS])).column_names

        mc_cats = json.loads(self._get_param(model_config, "cats", "string_value", default="[]"))
        mc_conts = json.loads(self._get_param(model_config, "conts", "string_value", default="[]"))

        self.cats = mc_cats or schema_cats
        self.conts = mc_conts or schema_conts

        workflow_outputs = set(workflow.output_schema.column_names)
        requested_cols = set(self.cats + self.conts)
        missing_cols = requested_cols - workflow_outputs
        extra_cols = workflow_outputs - requested_cols

        if missing_cols:
            raise ValueError(
                f"The following columns were not found in the workflow's output: {missing_cols}"
            )
        if extra_cols:
            raise ValueError(
                f"The following extra columns were found in the workflow's output: {extra_cols}"
            )

        # recurse over all column groups, initializing operators for inference pipeline
        self._initialize_ops(self.workflow.output_node)

    def _initialize_ops(self, workflow_node, visited=None):
        if visited is None:
            visited = set()

        if workflow_node.op and hasattr(workflow_node.op, "inference_initialize"):
            inference_op = workflow_node.op.inference_initialize(
                workflow_node.selector, self.model_config
            )
            if inference_op:
                workflow_node.op = inference_op

            supported = workflow_node.op.supports

            # if we're running on the CPU only, mask off support for GPU data formats
            if self.device == "CPU":
                supported = functools.reduce(
                    lambda a, b: a | b,
                    (v for v in list(Supports) if v & supported and "CPU" in str(v)),
                )
            # the 'supports' property is readonly, and we can't always attach a new property
            # to some of the operators (C++ categorify etc). set on the workflow_node instead
            workflow_node.inference_supports = supported

        for parent in workflow_node.parents_with_dependencies:
            if parent not in visited:
                visited.add(parent)
                self._initialize_ops(parent, visited)

    def run_workflow(self, input_tensors):
        # use our NVTabular workflow to transform the dataset
        transformed, kind = self._transform_tensors(input_tensors, self.workflow.output_node)

        # if we don't have tensors in numpy format, convert back so that the we can return
        # to triton
        if kind != Supports.CPU_DICT_ARRAY:
            transformed, kind = convert_format(transformed, kind, Supports.CPU_DICT_ARRAY)

        # convert to the format expected by the DL models
        return self._transform_outputs(transformed)

    @abstractmethod
    def _transform_outputs(self, tensors):
        pass

    def _convert_to_np(self, columns, tensors, dtype, rows):
        """converts outputs to a numpy input compatible with pytorch"""
        d = np.empty((rows, len(columns)), dtype=dtype)
        for i, name in enumerate(columns):
            d[:, i] = tensors[name].astype(dtype)
        return d

    def _transform_tensors(self, input_tensors, workflow_node):
        upstream_inputs = []

        # Gather inputs from the parents and dependency nodes
        if workflow_node.parents_with_dependencies:
            for parent in workflow_node.parents_with_dependencies:
                upstream_tensors, upstream_kind = self._transform_tensors(input_tensors, parent)
                if upstream_tensors is not None and upstream_kind:
                    upstream_inputs.append((upstream_tensors, upstream_kind))

        # Gather additional input columns from the original input tensors
        if workflow_node.selector:
            selector_columns = workflow_node.selector.names
            to_remove = []
            for upstream_tensors, upstream_kind in upstream_inputs:
                for col in selector_columns:
                    if col in upstream_tensors:
                        to_remove.append(col)
            for col in set(to_remove):
                selector_columns.remove(col)

            if selector_columns:
                selected_tensors = {c: input_tensors[c] for c in selector_columns}
                selected_kinds = Supports.CPU_DICT_ARRAY
                upstream_inputs.append((selected_tensors, selected_kinds))

        # Standardize the formats
        tensors, kind = None, None
        for upstream_tensors, upstream_kind in upstream_inputs:
            if tensors is None:
                tensors, kind = upstream_tensors, upstream_kind
            else:
                if kind != upstream_kind:
                    # we have multiple different kinds of data here (dataframe/array on cpu/gpu)
                    # we need to convert to a common format here first before concatenating.
                    op = workflow_node.op
                    if op and hasattr(op, "inference_supports"):
                        target_kind = op.inference_supports
                    else:
                        target_kind = Supports.CPU_DICT_ARRAY
                    # note : the 2nd convert_format call needs to be stricter in what the kind is
                    # (exact match rather than a bitmask of values)
                    tensors, kind = convert_format(tensors, kind, target_kind)
                    upstream_tensors, _ = convert_format(upstream_tensors, upstream_kind, kind)

                tensors = self.concat_tensors([tensors, upstream_tensors], kind)

        # Run the transform
        if tensors is not None and kind and workflow_node.op:
            try:
                # if the op doesn't support the current kind - we need to convert
                if (
                    hasattr(workflow_node, "inference_supports")
                    and not workflow_node.inference_supports & kind
                ):
                    tensors, kind = convert_format(tensors, kind, workflow_node.inference_supports)

                tensors = workflow_node.op.transform(
                    workflow_node.input_columns,
                    tensors,
                )

            except Exception:
                LOG.exception("Failed to transform operator %s", workflow_node.op)
                raise

        return tensors, kind

    def concat_tensors(self, tensors, kind):
        if kind & (Supports.GPU_DATAFRAME | Supports.CPU_DATAFRAME):
            return concat_columns(tensors)
        else:
            output = tensors[0]
            for tensor in tensors[1:]:
                output.update(tensor)
            return output

    def _get_param(self, config, *args, default=None):
        config_element = config["parameters"]
        for key in args:
            config_element = config_element.get(key, {})
        return config_element or default
