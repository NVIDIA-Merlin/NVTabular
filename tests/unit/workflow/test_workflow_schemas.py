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
import pytest

import nvtabular
from merlin.core.dispatch import make_df
from merlin.dag import ColumnSelector
from merlin.schema import ColumnSchema, Schema, Tags
from nvtabular import Workflow, ops


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


def test_fit_schema_works_when_subtracting_column_names():
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


def test_fit_schema_works_when_subtracting_nodes():
    schema = Schema(["x", "y", "id"])

    cont_features = (
        ColumnSelector(["x", "y"])
        >> ops.FillMissing()
        >> ops.Clip(min_value=0)
        >> ops.LogOp
        >> ops.Normalize()
        >> ops.Rename(postfix="_renamed")
    )

    subtract_features = ["y"] >> ops.Rename(postfix="_renamed")

    workflow1 = Workflow(cont_features - subtract_features)
    workflow1.fit_schema(schema)

    assert workflow1.output_schema.column_names == ["x_renamed"]


def test_fit_schema_works_when_subtracting_missing_nodes():
    schema = Schema(["x", "y", "id", "baseball"])

    cont_features = (
        ColumnSelector(["x", "y"])
        >> ops.FillMissing()
        >> ops.Clip(min_value=0)
        >> ops.LogOp
        >> ops.Normalize()
        >> ops.Rename(postfix="_renamed")
    )

    subtract_features = ["y", "baseball"] >> ops.Rename(postfix="_renamed")

    workflow1 = Workflow(cont_features - subtract_features)
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


def test_fit_schema_works_with_grouped_node_inputs():
    schema = Schema(["x", "y", "cost"])

    cat_features = ColumnSelector(["x", "y", ("x", "y")]) >> ops.TargetEncoding("cost")

    workflow1 = Workflow(cat_features)
    workflow1.fit_schema(schema)

    assert sorted(workflow1.output_schema.column_names) == sorted(
        ["TE_x_cost", "TE_y_cost", "TE_x_y_cost"]
    )


def test_fit_schema_works_with_node_dependencies():
    schema = Schema(["x", "y", "cost"])

    cont_features = ColumnSelector(["cost"]) >> ops.Rename(postfix="_renamed")
    cat_features = ColumnSelector(["x", "y"]) >> ops.TargetEncoding(cont_features)

    workflow1 = Workflow(cat_features)
    workflow1.fit_schema(schema)

    assert workflow1.output_schema.column_names == ["TE_x_cost_renamed", "TE_y_cost_renamed"]


# initial column selector works with tags
# filter within the workflow by tags
# test tags correct at output
@pytest.mark.parametrize(
    "op",
    [
        ops.Bucketize([1]),
        ops.Rename(postfix="_trim"),
        ops.Categorify(),
        ops.Categorify(encode_type="combo"),
        ops.Clip(0),
        ops.DifferenceLag("col1"),
        ops.FillMissing(),
        ops.Groupby("col1"),
        ops.HashBucket(1),
        ops.HashedCross(1),
        ops.JoinGroupby("col1"),
        ops.ListSlice(0),
        ops.LogOp(),
        ops.Normalize(),
        ops.TargetEncoding("col1"),
    ],
)
def test_workflow_select_by_tags(op):
    schema1 = ColumnSchema("col1", tags=["b", "c", "d"])
    schema2 = ColumnSchema("col2", tags=["c", "d"])
    schema3 = ColumnSchema("col3", tags=["d"])
    schema = Schema([schema1, schema2, schema3])

    cont_features = ColumnSelector(tags=["c"]) >> op
    workflow = Workflow(cont_features)
    workflow.fit_schema(schema)

    output_cols = op.output_column_names(ColumnSelector(["col1", "col2"]))
    assert len(workflow.output_schema.column_names) == len(output_cols.names)


def test_collision_tags_workflow():
    df = make_df(
        {
            "user_id": [1, 2, 3, 4, 6, 8, 5, 3] * 10,
            "rating": [1.5, 2.5, 3.0, 4.0, 5.0, 2.0, 3.0, 1.0] * 10,
        }
    )
    dataset = nvtabular.Dataset(df)

    cat_features = ["user_id"] >> ops.Categorify()

    te_features = cat_features >> ops.TargetEncoding(["rating"], kfold=5, p_smooth=20)
    te_features_norm = te_features >> ops.NormalizeMinMax()

    workflow = Workflow(te_features_norm).fit(dataset)

    for col_schema in workflow.output_schema.column_schemas.values():
        assert Tags.CONTINUOUS in col_schema.tags
        assert Tags.CATEGORICAL not in col_schema.tags


def test_graph_column_mapping():
    input_columns = ["a", "b"]
    input_schema = Schema(input_columns)

    rename_op_1 = input_columns >> ops.Rename(postfix="_renamed")
    rename_op_2 = rename_op_1 >> ops.Rename(postfix="_again")
    workflow = Workflow(rename_op_2)
    workflow.fit_schema(input_schema)

    assert workflow.graph.column_mapping == {"a_renamed_again": ["a"], "b_renamed_again": ["b"]}


def test_graph_column_mapping_expansion():
    input_columns = ["a", "b"]
    input_schema = Schema(input_columns)

    col_sim_op = [input_columns] >> ops.ColumnSimilarity(["a"], ["b"])
    rename_op = col_sim_op >> ops.Rename(postfix="_renamed")
    workflow = Workflow(col_sim_op + rename_op)
    workflow.fit_schema(input_schema)

    assert workflow.graph.column_mapping == {"a_b_sim": ["a", "b"], "a_b_sim_renamed": ["a", "b"]}


def test_remove_columns_single_op():
    input_columns = ["a", "b", "c", "label"]
    input_schema = Schema(input_columns)

    workflow_ops = input_columns >> ops.Rename(postfix="_nvt")
    workflow = Workflow(workflow_ops)
    workflow.fit_schema(input_schema)

    workflow1 = workflow.remove_inputs(["label"])

    expected_schema_in = Schema(["a", "b", "c"])
    expected_schema_out = Schema(["a_nvt", "b_nvt", "c_nvt"])

    assert workflow1.graph.input_schema == expected_schema_in
    assert workflow1.graph.output_schema == expected_schema_out


def test_remove_columns():
    input_columns = ["a", "b", "c", "label"]
    input_schema = Schema(input_columns)

    workflow_ops = input_columns >> ops.Rename(postfix="_nvt") >> ops.Rename(postfix="_onemore")
    rename_ops = workflow_ops >> ops.Rename(postfix="_another")
    workflow = Workflow(rename_ops)
    workflow.fit_schema(input_schema)

    workflow1 = workflow.remove_inputs(["label"])

    expected_schema_out = Schema(
        ["a_nvt_onemore_another", "b_nvt_onemore_another", "c_nvt_onemore_another"]
    )

    assert workflow1.graph.input_schema == Schema(["a", "b", "c"])
    assert workflow1.graph.output_schema == expected_schema_out


def test_remove_columns_combine():
    input_columns = ["a", "b", "c", "d"]
    input_schema = Schema(input_columns)

    workflow_ops = (
        [["a", "b"], ["c", "d"]] >> ops.ColumnSimilarity(None) >> ops.Rename(postfix="_renamed")
    )
    workflow = Workflow(workflow_ops)
    workflow.fit_schema(input_schema)

    workflow1 = workflow.remove_inputs(["c", "d"])

    expected_schema_in = Schema(["a", "b"])
    expected_schema_out = Schema(["a_b_sim_renamed"])

    assert workflow1.graph.input_schema == expected_schema_in
    assert workflow1.graph.output_schema.column_names == expected_schema_out.column_names
