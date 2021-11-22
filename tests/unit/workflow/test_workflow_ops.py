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

from nvtabular import ops as workflow_ops
from nvtabular.graph import ops as graph_ops


def test_column_concat_op():
    node = "col1" >> workflow_ops.Operator()
    col_name = "col2"

    concat_node = node + col_name
    assert isinstance(concat_node.op, graph_ops.ConcatColumns)


def test_column_subtraction_op():
    node = ["col1", "col2"] >> workflow_ops.Operator()
    col_name = "col1"

    subtract_node = node - col_name
    assert isinstance(subtract_node.op, graph_ops.SubtractionOp)


def test_column_subset_op():
    node = ["col1", "col2"] >> workflow_ops.Operator()

    bracket_node = node["col1"]
    assert isinstance(bracket_node.op, graph_ops.SubsetColumns)
