#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import logging

import numpy as np

from nvtabular.graph.graph import Graph
from nvtabular.graph.node import postorder_iter_nodes
from nvtabular.graph.ops.selection import SelectionOp
from nvtabular.graph.schema import Schema
from nvtabular.inference.graph.ops.tensorflow import TensorflowOp
from nvtabular.inference.graph.ops.workflow import WorkflowOp

LOG = logging.getLogger("nvtabular")


class InferenceGraph(Graph):
    def fit_schema(self, input_schema: Schema) -> "Graph":
        super().fit_schema(input_schema)

        model_node = None
        closest_wf_node = None

        nodes_bottom_up = reversed(list(postorder_iter_nodes(self.output_node)))

        for node in nodes_bottom_up:
            # Check if the node is a TF model and if so save it for later
            if isinstance(node.op, TensorflowOp):
                model_node = node

            # If we haven't found a workflow upstream of the model on
            # this branch of the graph yet, then this is where we need
            # to copy the model's dtypes to (to make them match)
            if not closest_wf_node and isinstance(node.op, WorkflowOp):
                closest_wf_node = node
                if model_node:
                    closest_wf_node.match_descendant_dtypes(model_node)

                    # TODO: Find a better way to override the dtypes ()
                    for (
                        col_name,
                        col_schema,
                    ) in closest_wf_node.output_schema.column_schemas.items():
                        closest_wf_node.op.workflow.output_dtypes[col_name] = np.dtype(
                            col_schema.dtype.as_numpy_dtype
                        )

            # If we find a selection node with no parents, this is the
            # end of the branch, so we should reset our workflow tracker
            # so that workflows on the next branch also get the correct
            # types from the model
            if not node.parents and isinstance(node.op, SelectionOp):
                closest_wf_node = None

        return self
