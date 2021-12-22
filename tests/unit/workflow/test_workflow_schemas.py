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
import glob
from pathlib import Path

import pytest

from nvtabular import Dataset, Workflow, ops
from nvtabular.graph import ColumnSchema, ColumnSelector, Schema
from nvtabular.graph.schema_io.schema_writer_pbtxt import PbTxt_SchemaWriter


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


@pytest.mark.parametrize("engine", ["parquet"])
def test_schema_write_read_dataset(tmpdir, dataset, engine):
    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    norms = ops.Normalize()
    cat_features = cat_names >> ops.Categorify(cat_cache="host")
    cont_features = cont_names >> ops.FillMissing() >> ops.Clip(min_value=0) >> ops.LogOp >> norms

    workflow = Workflow(cat_features + cont_features + label_name)

    workflow.fit(dataset)
    workflow.transform(dataset).to_parquet(
        tmpdir,
        out_files_per_proc=10,
    )

    schema_path = Path(tmpdir)
    proto_schema = PbTxt_SchemaWriter._read(schema_path / "schema.pbtxt")
    new_dataset = Dataset(glob.glob(str(tmpdir) + "/*.parquet"))
    assert """name: "name-cat"\n    min: 0\n    max: 27\n""" in str(proto_schema)
    assert new_dataset.schema == workflow.output_schema
