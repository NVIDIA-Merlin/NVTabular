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
from merlin.schema import Tags


def get_embedding_sizes(source, output_dtypes=None):
    """Returns a dictionary of embedding sizes from a workflow or workflow_node

    Parameters
    ----------
    source : Workflow or ColumnSelector
        Either a nvtabular Workflow or ColumnSelector object that we should use to find
        embedding sizes
    output_dtypes : dict, optional
        Optional dictionary of column_name:dtype. If passing a workflow object dtypes
        will be read from the workflow. This is used to figure out which columns
        are multihot-categorical, which are split out by this function. If passed a workflow_node
        and this parameter isn't set, you won't have multihot columns returned separately
    """
    # TODO: do we need to distinguish multihot columns here?  (if so why? )

    # have to lazy import Workflow to avoid circular import errors
    from nvtabular.workflow import Workflow

    output_node = source.output_node if isinstance(source, Workflow) else source

    if isinstance(source, Workflow):
        output_dtypes = output_dtypes or source.output_dtypes
    else:
        # passed in a column group
        output_dtypes = output_dtypes or {}

    output = {}
    multihot_columns = set()
    cats_schema = output_node.output_schema.select_by_tag(Tags.CATEGORICAL)
    for col_name, col_schema in cats_schema.column_schemas.items():
        if col_schema.dtype and col_schema.is_list and col_schema.is_ragged:
            # multi hot so remove from output and add to multihot
            multihot_columns.add(col_name)

        embeddings_sizes = col_schema.properties.get("embedding_sizes", {})
        cardinality = embeddings_sizes["cardinality"]
        dimensions = embeddings_sizes["dimension"]
        output[col_name] = (cardinality, dimensions)

    # TODO: returning different return types like this (based off the presence
    # of multihot features) is pretty janky. fix.
    if not multihot_columns:
        return output

    single_hots = {k: v for k, v in output.items() if k not in multihot_columns}
    multi_hots = {k: v for k, v in output.items() if k in multihot_columns}
    return single_hots, multi_hots
