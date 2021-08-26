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

from nvtabular import ops
from nvtabular.ops import internal


def test_column_concat_op():
    node = "col1" >> ops.Operator()
    col_name = "col2"

    concat_node = node + col_name
    assert isinstance(concat_node.op, internal.ConcatColumns)


def test_column_subset_op():
    node = ["col1", "col2"] >> ops.Operator()
    col_name = "col1"

    subtract_node = node - col_name
    assert isinstance(subtract_node.op, internal.SubsetColumns)
