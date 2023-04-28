#
# Copyright (c) 2023, NVIDIA CORPORATION.
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

import nvtabular.ops as ops
from merlin.dag import ColumnSelector
from merlin.dag.ops.subgraph import Subgraph
from nvtabular import Workflow


@pytest.mark.parametrize("engine", ["parquet"])
def test_workflow_chaining_subgraphs(dataset, engine):
    cont_features_1 = (
        ColumnSelector(["x", "y"]) >> ops.FillMissing() >> ops.Clip(min_value=0) >> ops.LogOp
    )

    second_graph = "*" >> ops.Normalize() >> ops.Rename(postfix="_renamed")

    cont_features = cont_features_1 >> Subgraph("second", second_graph)

    workflow = Workflow(cont_features["x_renamed"])
    result = workflow.fit_transform(dataset)

    df = result.to_ddf().compute()

    assert workflow.output_schema.column_names == ["x_renamed"]
    assert "x_renamed" in df


@pytest.mark.parametrize("engine", ["parquet"])
def test_workflow_adding_subgraphs(dataset, engine):
    first_op = Subgraph("first", ("*" >> ops.Rename(postfix="_renamed")))
    second_op = Subgraph("second", ("*" >> ops.Rename(postfix="_renamed")))

    features = (["x"] >> first_op) + (["y"] >> second_op)

    workflow = Workflow(features)
    result = workflow.fit_transform(dataset)

    df = result.to_ddf().compute()

    assert workflow.output_schema.column_names == ["x_renamed", "y_renamed"]
    assert "x_renamed" in df
    assert "y_renamed" in df
