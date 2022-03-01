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
import pathlib

from merlin.dag import ColumnSelector
from merlin.schema import Schema

from nvtabular.inference.graph.ops.operator import InferenceOperator
from nvtabular.inference.triton.ensemble import _generate_nvtabular_config


class WorkflowOp(InferenceOperator):
    def __init__(
        self,
        workflow,
        sparse_max=None,
        max_batch_size=None,
        label_columns=None,
        model_framework=None,
        cats=None,
        conts=None,
    ):
        super().__init__()

        self.workflow = workflow
        self.sparse_max = sparse_max or {}
        self.max_batch_size = max_batch_size
        self.label_columns = label_columns or []
        self.model_framework = model_framework or ""
        self.cats = cats or []
        self.conts = conts or []

    def compute_output_schema(
        self, input_schema: Schema, col_selector: ColumnSelector, prev_output_schema: Schema = None
    ) -> Schema:
        return self.workflow.output_schema

    def export(self, path, input_schema, output_schema, node_id=None, version=1):
        """Create a directory inside supplied path based on our export name"""
        modified_workflow = self.workflow.remove_inputs(self.label_columns)

        node_name = f"{self.export_name}_{node_id}" if node_id is not None else self.export_name

        node_export_path = pathlib.Path(path) / node_name
        node_export_path.mkdir(exist_ok=True)

        workflow_export_path = node_export_path / str(version) / "workflow"
        modified_workflow.save(str(workflow_export_path))

        return _generate_nvtabular_config(
            modified_workflow,
            node_name,
            node_export_path,
            backend="nvtabular",
            sparse_max=self.sparse_max,
            max_batch_size=self.max_batch_size,
            cats=self.cats,
            conts=self.conts,
        )
