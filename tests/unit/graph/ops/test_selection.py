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

from nvtabular.graph import ColumnSchema, ColumnSelector, Schema
from nvtabular.graph.ops.selection import SelectionOp


@pytest.mark.parametrize("engine", ["parquet"])
def test_selection_transform(df):
    selector = ColumnSelector(["x", "y"])
    op = SelectionOp(selector)

    result_df = op.transform(ColumnSelector(), df)

    assert (result_df.columns == ["x", "y"]).all()


@pytest.mark.parametrize("engine", ["parquet"])
def test_selection_output_column_names(df):
    selector = ColumnSelector(["x", "y"])
    op = SelectionOp(selector)

    result_selector = op.output_column_names(ColumnSelector())

    assert result_selector.names == ["x", "y"]


@pytest.mark.parametrize("engine", ["parquet"])
def test_selection_output_schema(df):
    selector = ColumnSelector(["x", "y"])
    schema = Schema([ColumnSchema(col) for col in df.columns])
    op = SelectionOp(selector)

    result_schema = op.compute_output_schema(schema, ColumnSelector())

    assert result_schema.column_names == ["x", "y"]
