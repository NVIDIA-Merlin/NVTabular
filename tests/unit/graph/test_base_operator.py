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

import nvtabular as nvt
import nvtabular.ops as wf_ops
from nvtabular.graph.base_operator import BaseOperator
from nvtabular.graph.schema import Schema
from nvtabular.graph.selector import ColumnSelector


@pytest.mark.parametrize("engine", ["parquet"])
def test_graph_validates_schemas(dataset, engine):
    workflow_ops = ["a", "b", "c"] >> wf_ops.Rename(postfix="_nvt")
    workflow = nvt.Workflow(workflow_ops)

    with pytest.raises(ValueError) as exc_info:
        workflow.fit(dataset)

    assert "Missing column" in str(exc_info.value)


@pytest.mark.parametrize("engine", ["parquet"])
def test_compute_selector_validates_schemas(dataset, engine):
    op = BaseOperator()
    schema = Schema(["a", "b"])
    selector = ColumnSelector(["c"])

    with pytest.raises(ValueError) as exc_info:
        op.compute_selector(schema, selector, ColumnSelector(), ColumnSelector())

    assert "Missing column" in str(exc_info.value)


@pytest.mark.parametrize("engine", ["parquet"])
def test_compute_input_schema_validates_schemas(dataset, engine):
    op = BaseOperator()
    schema = Schema(["a", "b"])
    selector = ColumnSelector(["c"])

    with pytest.raises(ValueError) as exc_info:
        op.compute_input_schema(schema, Schema(), Schema(), selector)

    assert "Missing column" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        op.compute_input_schema(Schema(), schema, Schema(), selector)

    assert "Missing column" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        op.compute_input_schema(Schema(), Schema(), schema, selector)

    assert "Missing column" in str(exc_info.value)


@pytest.mark.parametrize("engine", ["parquet"])
def test_compute_output_schema_validates_schemas(dataset, engine):
    op = BaseOperator()
    schema = Schema(["a", "b"])
    selector = ColumnSelector(["c"])

    with pytest.raises(ValueError) as exc_info:
        op.compute_output_schema(schema, selector)

    assert "Missing column" in str(exc_info.value)
