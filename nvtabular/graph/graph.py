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

import copy
import logging
from collections import deque

from nvtabular.graph.node import (
    Node,
    _combine_schemas,
    iter_nodes,
    postorder_iter_nodes,
    preorder_iter_nodes,
)
from nvtabular.graph.schema import Schema
from nvtabular.graph.selector import ColumnSelector

LOG = logging.getLogger("nvtabular")


class Graph:
    def __init__(self, output_node: Node):
        self.output_node = output_node

    @property
    def input_dtypes(self):
        if self.input_schema:
            return {
                name: col_schema.dtype
                for name, col_schema in self.input_schema.column_schemas.items()
            }
        else:
            return {}

    @property
    def output_dtypes(self):
        if self.output_schema:
            return {
                name: col_schema.dtype
                for name, col_schema in self.output_schema.column_schemas.items()
            }
        else:
            return {}

    @property
    def column_mapping(self):
        nodes = preorder_iter_nodes(self.output_node)
        column_mapping = self.output_node.column_mapping
        for node in list(nodes)[1:]:
            node_map = node.column_mapping
            for output_col, input_cols in column_mapping.items():
                early_inputs = []
                for input_col in input_cols:
                    early_inputs += node_map.get(input_col, [input_col])
                column_mapping[output_col] = early_inputs

        return column_mapping

    def construct_schema(self, root_schema: Schema, preserve_dtypes=False) -> "Graph":
        nodes = list(postorder_iter_nodes(self.output_node))

        self._compute_node_schemas(root_schema, nodes, preserve_dtypes)
        self._validate_node_schemas(root_schema, nodes, preserve_dtypes)

        return self

    def _compute_node_schemas(self, root_schema, nodes, preserve_dtypes=False):
        for node in nodes:
            node.compute_schemas(root_schema, preserve_dtypes=preserve_dtypes)

    def _validate_node_schemas(self, root_schema, nodes, strict_dtypes=False):
        for node in nodes:
            node.validate_schemas(root_schema, strict_dtypes=strict_dtypes)

    @property
    def input_schema(self):
        # leaf_node input and output schemas are the same (aka selection)
        return _combine_schemas(self.leaf_nodes)

    @property
    def leaf_nodes(self):
        return [node for node in postorder_iter_nodes(self.output_node) if not node.parents]

    @property
    def output_schema(self):
        return self.output_node.output_schema

    def _input_columns(self):
        input_cols = []
        for node in iter_nodes([self.output_node]):
            upstream_output_cols = []

            for upstream_node in node.parents_with_dependencies:
                upstream_output_cols += upstream_node.output_columns.names

            upstream_output_cols = _get_unique(upstream_output_cols)
            input_cols += list(set(node.input_columns.names) - set(upstream_output_cols))

        return _get_unique(input_cols)


def _get_schemaless_nodes(nodes):
    schemaless_nodes = []
    for node in iter_nodes(nodes):
        if node.input_schema is None:
            schemaless_nodes.append(node)

    return set(schemaless_nodes)


def _get_ops_by_type(nodes, op_type):
    return set(node for node in iter_nodes(nodes) if isinstance(node.op, op_type))


def _get_unique(cols):
    # Need to preserve order in unique-column list
    return list({x: x for x in cols}.keys())


def _remove_columns(workflow, to_remove):
    """
    Removes columns from a workflow

    This method will recursively remove columns from a workflow,
    starting at the input to the workflow
    and then removing all children of that column.

    Parameters
    -----------
    workflow : Workflow
        The workflow to remove columns from
    to_remove : array_like
        A list of input column names to remove from the workflow

    Returns
    -------
    Workflow
        A copy of the workflow with the columns removed
    """
    new_workflow = copy.deepcopy(workflow)

    # starting at the leafs tickle down looking for columns to remove
    # when found remove but then must propagate the removal of any other
    # output columns derived from that column.
    to_check = deque([(node, to_remove) for node in new_workflow.graph.leaf_nodes])

    seen_nodes = []  # TODO: remove this (for debugging only)
    while to_check:
        node, columns_to_remove = to_check.popleft()
        seen_nodes.append((node, columns_to_remove))

        output_columns_to_remove = []

        # remove cols
        for output_col_name, input_col_list in node.column_mapping.items():
            for name_to_remove in set(columns_to_remove):
                if name_to_remove in input_col_list:
                    # Remove the column name_to_remove from the inputs
                    del node.input_schema.column_schemas[name_to_remove]

                    # Remove the column name_to_remove from the selector
                    if node.selector:
                        original_selector = node.selector
                        node.selector = original_selector.filter_columns(
                            ColumnSelector([name_to_remove])
                        )

                    # Remove columns derived from name_to_remove from the outputs
                    if output_col_name in node.output_schema.column_names:
                        del node.output_schema.column_schemas[output_col_name]

                    output_columns_to_remove.append(output_col_name)

            if output_columns_to_remove:
                for child in node.children:
                    to_check.append((child, output_columns_to_remove))

    return new_workflow
