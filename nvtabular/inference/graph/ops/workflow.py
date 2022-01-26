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

from nvtabular.graph.schema import Schema
from nvtabular.graph.selector import ColumnSelector
from nvtabular.inference.graph.ops.operator import InferenceOperator
from nvtabular.inference.triton.ensemble import _generate_nvtabular_config


class WorkflowOp(InferenceOperator):
    def __init__(
        self,
        workflow,
        name=None,
        sparse_max=None,
        max_batch_size=None,
        label_columns=None,
        model_framework=None,
        cats=None,
        conts=None,
    ):
        self.workflow = workflow
        self.sparse_max = sparse_max or {}
        self.name = name or self.__class__.__name__.lower()
        self.max_batch_size = max_batch_size
        self.label_columns = label_columns or []
        self.model_framework = model_framework or ""
        self.cats = cats or []
        self.conts = conts or []

    def compute_output_schema(
        self, input_schema: Schema, col_selector: ColumnSelector, prev_output_schema: Schema = None
    ) -> Schema:
        return self.workflow.output_schema

    def export(self, path, input_schema, output_schema, version=1):
        """Create a directory inside supplied path based on our export name"""
        new_dir_path = pathlib.Path(path) / self.export_name
        new_dir_path.mkdir()

        modified_workflow = self.workflow.remove_inputs(self.label_columns)

        # TODO: Extract this logic to base inference operator?
        export_path = new_dir_path / str(version) / "workflow"
        modified_workflow.save(str(export_path))

        return _generate_nvtabular_config(
            modified_workflow,
            self.export_name,
            new_dir_path,
            backend="nvtabular",
            sparse_max=self.sparse_max,
            max_batch_size=self.max_batch_size,
            cats=self.cats,
            conts=self.conts,
        )

    @property
    def export_name(self):
        return self.name
