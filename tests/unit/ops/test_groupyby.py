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
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

import nvtabular as nvt
import nvtabular.io
from nvtabular import ColumnSelector, Schema, Workflow, ops

try:
    import cudf

    _CPU = [True, False]
except ImportError:
    _CPU = [True]


@pytest.mark.parametrize("cpu", _CPU)
@pytest.mark.parametrize("keys", [["name"], "id", ["name", "id"]])
def test_groupby_op(keys, cpu):
    # Initial timeseries dataset
    size = 60
    df1 = nvt.dispatch._make_df(
        {
            "name": np.random.choice(["Dave", "Zelda"], size=size),
            "id": np.random.choice([0, 1], size=size),
            "ts": np.linspace(0.0, 10.0, num=size),
            "x": np.arange(size),
            "y": np.linspace(0.0, 10.0, num=size),
            "shuffle": np.random.uniform(low=0.0, high=10.0, size=size),
        }
    )
    df1 = df1.sort_values("shuffle").drop(columns="shuffle").reset_index(drop=True)

    # Create a ddf, and be sure to shuffle by the groupby keys
    ddf1 = dd.from_pandas(df1, npartitions=3).shuffle(keys)
    dataset = nvt.Dataset(ddf1, cpu=cpu)
    dataset.schema.column_schemas["x"] = (
        dataset.schema.column_schemas["name"].with_name("x").with_tags("custom_tag")
    )
    # Define Groupby Workflow
    groupby_features = ColumnSelector(["name", "id", "ts", "x", "y"]) >> ops.Groupby(
        groupby_cols=keys,
        sort_cols=["ts"],
        aggs={
            "x": ["list", "sum"],
            "y": ["first", "last"],
            "ts": ["min"],
        },
        name_sep="-",
    )
    processor = nvtabular.Workflow(groupby_features)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()

    assert "custom_tag" in processor.output_schema.column_schemas["x-list"].tags

    if not cpu:
        # Make sure we are capturing the list type in `output_dtypes`
        assert processor.output_dtypes["x-list"] == cudf.core.dtypes.ListDtype("int64")

    # Check list-aggregation ordering
    x = new_gdf["x-list"]
    x = x.to_pandas() if hasattr(x, "to_pandas") else x
    sums = []
    for el in x.values:
        _el = pd.Series(el)
        sums.append(_el.sum())
        assert _el.is_monotonic_increasing

    # Check that list sums match sum aggregation
    x = new_gdf["x-sum"]
    x = x.to_pandas() if hasattr(x, "to_pandas") else x
    assert list(x) == sums

    # Check basic behavior or "y" column
    assert (new_gdf["y-first"] < new_gdf["y-last"]).all()


def test_groupby_selector_cols():
    input_schema = Schema(["name", "id", "ts", "x", "y"])

    # Include the groupby col in the selector
    groupby = ColumnSelector(["name", "id", "ts", "x", "y"]) >> ops.Groupby(
        groupby_cols=["name"],
        sort_cols=["ts"],
        aggs={
            "x": ["list", "sum"],
            "y": ["first", "last"],
            "ts": ["min"],
        },
        name_sep="-",
    )
    workflow = Workflow(groupby).fit_schema(input_schema)

    # If groupby_cols are included in the selector, they should be in the output
    assert "name" in workflow.output_node.output_schema.column_names

    # Don't include the groupby col in the selector
    groupby = ColumnSelector(["id", "ts", "x", "y"]) >> ops.Groupby(
        groupby_cols=["name"],
        sort_cols=["ts"],
        aggs={
            "x": ["list", "sum"],
            "y": ["first", "last"],
            "ts": ["min"],
        },
        name_sep="-",
    )
    workflow = Workflow(groupby).fit_schema(input_schema)

    # If groupby_cols aren't included in the selector, they shouldn't be in the output
    assert "name" not in workflow.output_node.output_schema.column_names
