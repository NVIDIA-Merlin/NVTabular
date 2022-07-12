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

import pandas as pd
from dask.core import flatten

from merlin.core.dispatch import concat_columns, is_list_dtype, list_val_dtype
from merlin.dag import Node

LOG = logging.getLogger("nvtabular")


class MerlinPythonExecutor:
    def apply(self, root_df, workflow_nodes, additional_columns=None, capture_dtypes=False):
        """Transforms a single partition by appyling all operators in a WorkflowNode"""
        output = None

        for node in workflow_nodes:
            node_input_cols = get_unique(node.input_schema.column_names)
            node_output_cols = get_unique(node.output_schema.column_names)
            addl_input_cols = set(node.dependency_columns.names)

            # Build input dataframe
            if node.parents_with_dependencies:
                # If there are parents, collect their outputs
                # to build the current node's input
                input_df = None
                seen_columns = None

                for parent in node.parents_with_dependencies:
                    parent_output_cols = get_unique(parent.output_schema.column_names)
                    parent_df = self.apply(root_df, [parent], capture_dtypes=capture_dtypes)
                    if input_df is None or not len(input_df):
                        input_df = parent_df[parent_output_cols]
                        seen_columns = set(parent_output_cols)
                    else:
                        new_columns = set(parent_output_cols) - seen_columns
                        input_df = concat_columns([input_df, parent_df[list(new_columns)]])
                        seen_columns.update(new_columns)

                # Check for additional input columns that aren't generated by parents
                # and fetch them from the root dataframe
                unseen_columns = set(node.input_schema.column_names) - seen_columns
                addl_input_cols = addl_input_cols.union(unseen_columns)

                # TODO: Find a better way to remove dupes
                addl_input_cols = addl_input_cols - set(input_df.columns)

                if addl_input_cols:
                    input_df = concat_columns([input_df, root_df[list(addl_input_cols)]])
            else:
                # If there are no parents, this is an input node,
                # so pull columns directly from root df
                input_df = root_df[node_input_cols + list(addl_input_cols)]

            # Compute the node's output
            if node.op:
                try:
                    # use input_columns to ensure correct grouping (subgroups)
                    selection = node.input_columns.resolve(node.input_schema)
                    output_df = node.op.transform(selection, input_df)

                    # Update or validate output_df dtypes
                    for col_name, output_col_schema in node.output_schema.column_schemas.items():
                        col_series = output_df[col_name]
                        col_dtype = col_series.dtype
                        is_list = is_list_dtype(col_series)

                        if is_list:
                            col_dtype = list_val_dtype(col_series)

                        output_df_schema = output_col_schema.with_dtype(
                            col_dtype, is_list=is_list, is_ragged=is_list
                        )

                        if capture_dtypes:
                            node.output_schema.column_schemas[col_name] = output_df_schema
                        elif len(output_df):
                            if output_col_schema.dtype != output_df_schema.dtype:
                                raise TypeError(
                                    f"Dtype discrepancy detected for column {col_name}: "
                                    f"operator {node.op.label} reported dtype "
                                    f"`{output_col_schema.dtype}` but returned dtype "
                                    f"`{output_df_schema.dtype}`."
                                )
                except Exception:
                    LOG.exception("Failed to transform operator %s", node.op)
                    raise
                if output_df is None:
                    raise RuntimeError(f"Operator {node.op} didn't return a value during transform")
            else:
                output_df = input_df

            # Combine output across node loop iterations

            # dask needs output to be in the same order defined as meta, reorder partitions here
            # this also selects columns (handling the case of removing columns from the output using
            # "-" overload)
            if output is None:
                output = output_df[node_output_cols]
            else:
                output = concat_columns([output, output_df[node_output_cols]])

        if additional_columns:
            output = concat_columns([output, root_df[get_unique(additional_columns)]])

        return output


class MerlinDaskExecutor:
    def __init__(self):
        self._executor = MerlinPythonExecutor()

    def apply(self, ddf, nodes, output_dtypes=None, additional_columns=None, capture_dtypes=False):
        """
        Transforms all partitions of a Dask Dataframe by applying the operators
        from a collection of Nodes
        """

        # Check if we are only selecting columns (no transforms).
        # If so, we should perform column selection at the ddf level.
        # Otherwise, Dask will not push the column selection into the
        # IO function.
        if not nodes:
            return ddf[get_unique(additional_columns)] if additional_columns else ddf

        if isinstance(nodes, Node):
            nodes = [nodes]

        columns = list(flatten(wfn.output_columns.names for wfn in nodes))
        columns += additional_columns if additional_columns else []

        if isinstance(output_dtypes, dict) and isinstance(ddf._meta, pd.DataFrame):
            dtypes = output_dtypes
            output_dtypes = type(ddf._meta)({k: [] for k in columns})
            for column, dtype in dtypes.items():
                output_dtypes[column] = output_dtypes[column].astype(dtype)

        elif not output_dtypes:
            # TODO: constructing meta like this loses dtype information on the ddf
            # and sets it all to 'float64'. We should propagate dtype information along
            # with column names in the columngroup graph. This currently only
            # happesn during intermediate 'fit' transforms, so as long as statoperators
            # don't require dtype information on the DDF this doesn't matter all that much
            output_dtypes = type(ddf._meta)({k: [] for k in columns})

        return ddf.map_partitions(
            self._executor.apply,
            nodes,
            additional_columns=additional_columns,
            capture_dtypes=capture_dtypes,
            meta=output_dtypes,
            enforce_metadata=False,
        )


def get_unique(cols):
    # Need to preserve order in unique-column list
    return list({x: x for x in cols}.keys())
