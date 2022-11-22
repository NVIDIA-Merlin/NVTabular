#
# Copyright (c) 2022, NVIDIA CORPORATION.
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
import numpy as np
import pytest

from merlin.core.dispatch import make_df
from merlin.io import Dataset
from merlin.schema import ColumnSchema, Tags
from nvtabular import Workflow
from nvtabular.ops import ValueCount


@pytest.mark.parametrize(
    ["values", "expected_col_schema"],
    [
        [
            [[1, 2, 3], [4, 5, 6]],
            ColumnSchema(
                "feature",
                dtype=np.int64,
                is_ragged=False,
                properties={"value_count": {"min": 3, "max": 3}},
            ),
        ],
        [
            [[1, 2, 3], [4, 5], [6]],
            ColumnSchema(
                "feature",
                dtype=np.int64,
                is_ragged=True,
                properties={"value_count": {"min": 1, "max": 3}},
            ),
        ],
    ],
)
def test_value_count_schema(values, expected_col_schema):
    df = make_df({"feature": values})
    dataset = Dataset(df)

    workflow = Workflow(["feature"] >> ValueCount())
    workflow.fit(dataset)

    transformed = workflow.transform(dataset)

    assert transformed.schema["feature"] == expected_col_schema.with_name("feature")


def test_value_count_multiple_partitions():
    df = make_df({"feature": ["1", "2", "3"], "session": [1, 1, 2]})
    dataset = Dataset(df, npartitions=1)

    workflow = Workflow(["feature"] >> ValueCount())
    workflow.fit(dataset)

    transformed = workflow.transform(dataset)

    expected_col_schema = ColumnSchema(
        "feature",
        dtype=np.int64,
        is_ragged=False,
        tags=[Tags.LIST],
        properties={"value_count": {"min": 2, "max": 2}},
    )

    assert transformed.schema["feature"] == expected_col_schema.with_name("feature")
