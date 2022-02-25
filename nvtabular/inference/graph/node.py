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
from merlin.dag import Node
from merlin.schema import Schema


class InferenceNode(Node):
    def export(self, output_path, node_id=None, version=1):
        return self.op.export(
            output_path, self.input_schema, self.output_schema, node_id=node_id, version=version
        )

    @property
    def export_name(self):
        return self.op.export_name

    @property
    def exportable(self):
        return hasattr(self.op, "export")

    def match_descendant_dtypes(self, source_node):
        self.output_schema = _match_dtypes(source_node.input_schema, self.output_schema)
        return self

    def match_ancestor_dtypes(self, source_node):
        self.input_schema = _match_dtypes(source_node.output_schema, self.input_schema)
        return self


def _match_dtypes(source_schema, dest_schema):
    matched = Schema()
    for col_name, col_schema in dest_schema.column_schemas.items():
        source_dtype = source_schema.get(col_name, col_schema).dtype
        matched[col_name] = col_schema.with_dtype(source_dtype)

    return matched
