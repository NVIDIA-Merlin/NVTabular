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
import pandas as pd
import pytest

import nvtabular as nvt
from nvtabular import ops

try:
    import dask_cudf

    _CPU = [True, False]
    _HAS_GPU = True
except ImportError:
    _CPU = [True]
    _HAS_GPU = False


@pytest.mark.parametrize("cpu", _CPU)
def test_joingroupby_dependency(tmpdir, cpu):
    df = pd.DataFrame(
        {
            "Author": ["User_A", "User_A", "User_A", "User_B", "User_B"],
            "Cost": [100.0, 200.0, 300.0, 400.0, 400.0],
        }
    )

    normalized_cost = ["Cost"] >> nvt.ops.NormalizeMinMax() >> nvt.ops.Rename(postfix="_normalized")
    groupby_features = ["Author"] >> ops.JoinGroupby(
        out_path=str(tmpdir), stats=["sum"], cont_cols=normalized_cost
    )
    workflow = nvt.Workflow(groupby_features)

    df_out = workflow.fit_transform(nvt.Dataset(df, cpu=cpu)).to_ddf().compute()
    if cpu:
        assert df_out["Author_Cost_normalized_sum"].to_list() == [1.0, 1.0, 1.0, 2.0, 2.0]
    else:
        assert df_out["Author_Cost_normalized_sum"].to_arrow().to_pylist() == [
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
        ]


@pytest.mark.parametrize("cpu", _CPU)
@pytest.mark.parametrize("groups", [[["Author", "Engaging-User"]], "Author"])
def test_joingroupby_multi(tmpdir, groups, cpu):
    df = pd.DataFrame(
        {
            "Author": ["User_A", "User_A", "User_A", "User_B"],
            "Engaging-User": ["User_B", "User_B", "User_C", "User_C"],
            "Cost": [100.0, 200.0, 300.0, 400.0],
            "Post": [1, 2, 3, 4],
        }
    )

    groupby_features = groups >> ops.JoinGroupby(
        out_path=str(tmpdir), stats=["sum"], cont_cols=["Cost"]
    )
    workflow = nvt.Workflow(groupby_features + "Post")

    df_out = workflow.fit_transform(nvt.Dataset(df, cpu=cpu)).to_ddf().compute()

    if isinstance(groups, list):
        # Join on ["Author", "Engaging-User"]
        if cpu:
            check = df_out["Author_Engaging-User_Cost_sum"].to_list()
        else:
            check = df_out["Author_Engaging-User_Cost_sum"].to_arrow().to_pylist()
        assert check == [300.0, 300.0, 300.0, 400.0]
    else:
        # Join on ["Author"]
        if cpu:
            check = df_out["Author_Cost_sum"].to_list()
        else:
            check = df_out["Author_Cost_sum"].to_arrow().to_pylist()
        assert check == [600.0, 600.0, 600.0, 400.0]


@pytest.mark.skipif(not _HAS_GPU, reason="This unittest requires cudf/dask_cudf to run")
@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize(
    "kind_ext",
    [
        "cudf",
        "pandas",
        "arrow",
        "parquet",
        "parquet-multi",
        "csv",
        "dask-dataframe",
        "dask-cudf",
        "dataset",
    ],
)
@pytest.mark.parametrize("cache", ["host", "device"])
@pytest.mark.parametrize("how", ["left", "inner"])
@pytest.mark.parametrize("cpu", _CPU)
@pytest.mark.parametrize("drop_duplicates", [True, False])
def test_join_external(tmpdir, df, dataset, engine, kind_ext, cache, how, cpu, drop_duplicates):
    # Define "external" table
    shift = 100
    df_ext = df[["id"]].copy().sort_values("id")
    df_ext["new_col"] = df_ext["id"] + shift
    df_ext["new_col_2"] = "keep"
    df_ext["new_col_3"] = "ignore"
    df_ext_check = df_ext.copy()
    if kind_ext == "pandas":
        df_ext = df_ext.to_pandas()
    elif kind_ext == "arrow":
        df_ext = df_ext.to_arrow()
    elif kind_ext == "parquet":
        path = tmpdir.join("external.parquet")
        df_ext.to_parquet(path)
        df_ext = path
    elif kind_ext == "parquet-multi":
        path = tmpdir.join("external-multi.parquet")
        dask_cudf.from_cudf(df_ext, npartitions=3).to_parquet(path)
        df_ext = path
    elif kind_ext == "csv":
        path = tmpdir.join("external.csv")
        df_ext.to_csv(path)
        df_ext = path
    elif kind_ext == "dask-dataframe":
        df_ext = dd.from_pandas(df_ext.to_pandas(), npartitions=2)
    elif kind_ext == "dask-cudf":
        df_ext = dask_cudf.from_cudf(df_ext, npartitions=2)
    elif kind_ext == "dataset":
        df_ext = nvt.Dataset(df_ext)

    # Define Op
    on = "id"
    columns_left = list(df.columns)
    columns_ext = ["id", "new_col", "new_col_2"]
    df_ext_check = df_ext_check[columns_ext]
    if drop_duplicates:
        df_ext_check.drop_duplicates(ignore_index=True, inplace=True)
    joined = nvt.ColumnSelector(columns_left) >> nvt.ops.JoinExternal(
        df_ext,
        on,
        how=how,
        columns_ext=columns_ext,
        cache=cache,
        drop_duplicates_ext=drop_duplicates,
    )

    gdf = df.reset_index()
    dataset = nvt.Dataset(gdf, cpu=cpu)
    processor = nvt.Workflow(joined)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute().reset_index()

    check_gdf = gdf.merge(df_ext_check, how=how, on=on)
    assert len(check_gdf) == len(new_gdf)
    assert (new_gdf["id"] + shift).all() == new_gdf["new_col"].all()
    assert gdf["id"].all() == new_gdf["id"].all()
    assert "new_col_2" in new_gdf.columns
    assert "new_col_3" not in new_gdf.columns
