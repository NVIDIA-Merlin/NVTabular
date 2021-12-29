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
from nvtabular.graph import Node
from nvtabular.graph.schema import Schema


class InferenceNode(Node):
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
