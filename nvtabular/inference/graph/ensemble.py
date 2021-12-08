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
from nvtabular.graph.graph import Graph
from nvtabular.graph.ops.selection import SelectionOp
from nvtabular.inference.graph.ops.tensorflow import TensorflowOp
from nvtabular.inference.graph.ops.workflow import WorkflowOp
from nvtabular.inference.triton.ensemble import export_tensorflow_ensemble  # noqa


class Ensemble:
    def __init__(self, ops, schema, name="ensemble_model", label_columns=None):
        self.graph = Graph(ops)
        self.graph.fit_schema(schema)
        self.name = name
        self.label_columns = label_columns or []

    def export(self, export_path):
        parents = self.graph.output_node.parents_with_dependencies
        assert len(parents) == 1

        workflow_node = parents[0]
        assert len(workflow_node.parents_with_dependencies) == 1
        assert isinstance(workflow_node.op, WorkflowOp)

        selection_node = workflow_node.parents_with_dependencies[0]
        assert len(selection_node.parents_with_dependencies) == 0
        assert isinstance(selection_node.op, SelectionOp)

        model_node = self.graph.output_node
        assert isinstance(model_node.op, TensorflowOp)

        export_tensorflow_ensemble(
            model_node.op.model,
            workflow_node.op.workflow,
            self.name,
            export_path,
            self.label_columns,
        )
