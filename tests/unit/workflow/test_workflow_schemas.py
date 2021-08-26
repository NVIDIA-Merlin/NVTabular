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

from nvtabular import Workflow, ops
from nvtabular.columns import ColumnSelector, Schema


def test_fit_schema():
    schema = Schema(["x", "y", "id"])

    cont_features = (
        ColumnSelector(schema.column_names)
        >> ops.FillMissing()
        >> ops.Clip(min_value=0)
        >> ops.LogOp
        >> ops.Normalize()
        >> ops.Rename(postfix="_renamed")
    )

    workflow = Workflow(cont_features)
    workflow.fit_schema(schema)

    assert workflow.output_schema.column_names == ["x_renamed", "y_renamed", "id_renamed"]


def test_fit_schema_works_with_addition_nodes():
    schema = Schema(["x", "y", "id"])

    x_node = ColumnSelector(["x"]) >> ops.Rename(postfix="_renamed")

    workflow = Workflow(x_node + "y")
    workflow.fit_schema(schema)

    assert workflow.output_schema.column_names == ["x_renamed", "y"]

    x_node = ColumnSelector(["x"]) >> ops.Rename(postfix="_renamed")
    y_node = ColumnSelector(["y"]) >> ops.Rename(postfix="_renamed")

    workflow = Workflow(x_node + y_node)
    workflow.fit_schema(schema)

    assert workflow.output_schema.column_names == ["x_renamed", "y_renamed"]


def test_fit_schema_works_with_subtraction_nodes():
    schema = Schema(["x", "y", "id"])

    cont_features = (
        ColumnSelector(["x", "y"])
        >> ops.FillMissing()
        >> ops.Clip(min_value=0)
        >> ops.LogOp
        >> ops.Normalize()
        >> ops.Rename(postfix="_renamed")
    )

    workflow1 = Workflow(cont_features - "y_renamed")
    workflow1.fit_schema(schema)

    assert workflow1.output_schema.column_names == ["x_renamed"]


def test_fit_schema_works_with_selection_nodes():
    schema = Schema(["x", "y", "id"])

    cont_features = (
        ColumnSelector(["x", "y"])
        >> ops.FillMissing()
        >> ops.Clip(min_value=0)
        >> ops.LogOp
        >> ops.Normalize()
        >> ops.Rename(postfix="_renamed")
    )

    workflow = Workflow(cont_features["x_renamed"])
    workflow.fit_schema(schema)

    assert workflow.output_schema.column_names == ["x_renamed"]


def test_fit_schema_works_with_raw_column_dependencies():
    schema = Schema(["x", "y", "cost"])

    cat_features = ColumnSelector(["x", "y"]) >> ops.TargetEncoding("cost")

    workflow = Workflow(cat_features)
    workflow.fit_schema(schema)

    assert workflow.output_schema.column_names == ["TE_x_cost", "TE_y_cost"]


def test_fit_schema_works_with_node_dependencies():
    schema = Schema(["x", "y", "cost"])

    cont_features = ColumnSelector(["cost"]) >> ops.Rename(postfix="_renamed")
    cat_features = ColumnSelector(["x", "y"]) >> ops.TargetEncoding(cont_features)

    workflow1 = Workflow(cat_features)
    workflow1.fit_schema(schema)

    assert workflow1.output_schema.column_names == ["TE_x_cost_renamed", "TE_y_cost_renamed"]
