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
import copy

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

import nvtabular as nvt
import nvtabular.io
from nvtabular import ColumnSelector, dispatch, ops
from tests.conftest import assert_eq, mycols_csv, mycols_pq

try:
    import cudf
    import dask_cudf

    _CPU = [True, False]
    _HAS_GPU = True
except ImportError:
    _CPU = [True]
    _HAS_GPU = False


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1] if _HAS_GPU else [None])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("op_columns", [["x"], ["x", "y"]])
@pytest.mark.parametrize("cpu", _CPU)
def test_log(tmpdir, df, dataset, gpu_memory_frac, engine, op_columns, cpu):
    cont_features = op_columns >> nvt.ops.LogOp()
    processor = nvt.Workflow(cont_features)
    processor.fit(dataset)
    new_df = processor.transform(dataset).to_ddf().compute()
    for col in op_columns:
        values = dispatch._array(new_df[col])
        original = dispatch._array(df[col])
        assert_eq(values, np.log(original.astype(np.float32) + 1))


@pytest.mark.parametrize("cpu", _CPU)
def test_logop_lists(tmpdir, cpu):
    df = dispatch._make_df(device="cpu" if cpu else "gpu")
    df["vals"] = [[np.exp(0) - 1, np.exp(1) - 1], [np.exp(2) - 1], []]

    features = ["vals"] >> nvt.ops.LogOp()
    workflow = nvt.Workflow(features)
    new_df = workflow.fit_transform(nvt.Dataset(df)).to_ddf().compute()

    expected = dispatch._make_df(device="cpu" if cpu else "gpu")
    expected["vals"] = [[0.0, 1.0], [2.0], []]

    assert_eq(expected, new_df)


def test_valuecount(tmpdir):
    df = dispatch._make_df(
        {
            "list1": [[1, 2, 3, 4], [3, 2, 1], [1, 4], [0]],
            "list2": [[1, 4], [3, 2, 1], [0, 4], [1, 4, 5]],
        }
    )
    ds = nvt.Dataset(df)
    val_count = nvt.ops.ValueCount()
    feats = ["list1", "list2"] >> val_count
    feats1 = feats["list1"] >> nvt.ops.AddMetadata(tags=["categorical"])
    feats2 = feats["list2"] >> nvt.ops.AddMetadata(tags=["continuous"])
    processor = nvt.Workflow(feats1 + feats2)
    processor.fit(ds)
    processor.transform(ds).to_parquet(tmpdir, out_files_per_proc=1)
    assert "list1" in list(val_count.stats.keys())
    assert "list2" in list(val_count.stats.keys())
    new_df = nvt.Dataset(str(tmpdir), engine="parquet")
    assert processor.output_schema.column_schemas["list1"].properties == {
        "value_count": {"min": 1, "max": 4}
    }
    assert processor.output_schema.column_schemas["list2"].properties == {
        "value_count": {"min": 2, "max": 3}
    }
    assert new_df.schema.column_schemas["list1"].properties == {"value_count": {"min": 1, "max": 4}}
    assert new_df.schema.column_schemas["list2"].properties == {"value_count": {"min": 2, "max": 3}}

    assert new_df.schema.column_schemas["list1"].tags == [nvt.graph.tags.Tags.CATEGORICAL]
    assert new_df.schema.column_schemas["list2"].tags == [nvt.graph.tags.Tags.CONTINUOUS]


@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("cpu", _CPU)
def test_dropna(tmpdir, df, dataset, engine, cpu):
    columns = mycols_pq if engine == "parquet" else mycols_csv
    dropna_features = columns >> ops.Dropna()
    if cpu:
        dataset.to_cpu()

    processor = nvt.Workflow(dropna_features)
    processor.fit(dataset)

    new_df = processor.transform(dataset).to_ddf().compute()
    assert new_df.columns.all() == df.columns.all()
    assert new_df.isnull().all().sum() < 1, "null values exist"


@pytest.mark.parametrize("cpu", _CPU)
@pytest.mark.parametrize("gpu_memory_frac", [0.1])
@pytest.mark.parametrize("engine", ["parquet"])
def test_filter(tmpdir, df, dataset, gpu_memory_frac, engine, cpu):
    if cpu and not isinstance(df, pd.DataFrame):
        df = df.to_pandas()

    cont_names = ["x", "y"]
    filtered = cont_names >> ops.Filter(f=lambda df: df[df["y"] > 0.5])
    processor = nvtabular.Workflow(filtered)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute().reset_index()
    filter_df = df[df["y"] > 0.5].reset_index()
    for col in cont_names:
        assert np.all((new_gdf[col] - filter_df[col]).abs().values <= 1e-2)

    # return isnull() rows
    for col in cont_names:
        idx = np.random.choice(df.shape[0] - 1, int(df.shape[0] * 0.2))
        df[col].iloc[idx] = None

    filtered = cont_names >> ops.Filter(f=lambda df: df[df.x.isnull()])
    processor = nvtabular.Workflow(filtered)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()
    assert new_gdf.shape[0] < df.shape[0], "null values do not exist"

    # again testing filtering by returning a series rather than a df
    filtered = cont_names >> ops.Filter(f=lambda df: df.x.isnull())
    processor = nvtabular.Workflow(filtered)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()
    assert new_gdf.shape[0] < df.shape[0], "null values do not exist"

    # if the filter returns an invalid type we should get an exception immediately
    # (rather than causing problems downstream in the workflow)
    filtered = cont_names >> ops.Filter(f=lambda df: "some invalid value")
    processor = nvtabular.Workflow(filtered)
    with pytest.raises(ValueError):
        new_gdf = processor.transform(dataset).to_ddf().compute()


@pytest.mark.parametrize("cpu", _CPU)
def test_difference_lag(cpu):
    lib = pd if cpu else cudf
    df = lib.DataFrame(
        {"userid": [0, 0, 0, 1, 1, 2], "timestamp": [1000, 1005, 1100, 2000, 2001, 3000]}
    )

    diff_features = ["timestamp"] >> ops.DifferenceLag(partition_cols=["userid"], shift=[1, -1])
    dataset = nvt.Dataset(df, cpu=cpu)
    processor = nvtabular.Workflow(diff_features)
    processor.fit(dataset)
    new_df = processor.transform(dataset).to_ddf().compute()

    assert new_df["timestamp_difference_lag_1"][1] == 5
    assert new_df["timestamp_difference_lag_1"][2] == 95
    if cpu:
        assert lib.isna(new_df["timestamp_difference_lag_1"][0])
        assert lib.isna(new_df["timestamp_difference_lag_1"][3])
    else:
        assert new_df["timestamp_difference_lag_1"][0] is (lib.NA if hasattr(lib, "NA") else None)
        assert new_df["timestamp_difference_lag_1"][3] is (lib.NA if hasattr(lib, "NA") else None)

    assert new_df["timestamp_difference_lag_-1"][0] == -5
    assert new_df["timestamp_difference_lag_-1"][1] == -95
    assert new_df["timestamp_difference_lag_-1"][3] == -1
    if cpu:
        assert lib.isna(new_df["timestamp_difference_lag_-1"][2])
        assert lib.isna(new_df["timestamp_difference_lag_-1"][5])
    else:
        assert new_df["timestamp_difference_lag_-1"][2] is (lib.NA if hasattr(lib, "NA") else None)
        assert new_df["timestamp_difference_lag_-1"][5] is (lib.NA if hasattr(lib, "NA") else None)


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1] if _HAS_GPU else [None])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("cpu", _CPU)
def test_hashed_cross(tmpdir, df, dataset, gpu_memory_frac, engine, cpu):
    # TODO: add tests for > 2 features, multiple crosses, etc.
    cat_names = [["name-string", "id"]]
    num_buckets = 10

    hashed_cross = cat_names >> ops.HashedCross(num_buckets)
    dataset = nvt.Dataset(df, cpu=cpu)
    processor = nvtabular.Workflow(hashed_cross)
    processor.fit(dataset)
    new_df = processor.transform(dataset).to_ddf().compute()

    # check sums for determinancy
    new_column_name = "_X_".join(cat_names[0])
    assert np.all(new_df[new_column_name].values >= 0)
    assert np.all(new_df[new_column_name].values <= 9)
    checksum = new_df[new_column_name].sum()
    new_df = processor.transform(dataset).to_ddf().compute()
    assert new_df[new_column_name].sum() == checksum


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1] if _HAS_GPU else [None])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("cpu", _CPU)
def test_bucketized(tmpdir, df, dataset, gpu_memory_frac, engine, cpu):
    cont_names = ["x", "y"]
    boundaries = [[-1, 0, 1], [-4, 100]]

    bucketize_op = ops.Bucketize(dict(zip(cont_names, boundaries)))

    bucket_features = cont_names >> bucketize_op
    processor = nvtabular.Workflow(bucket_features)

    ds = copy.copy(dataset)
    if cpu:
        ds.to_cpu()
    processor.fit(ds)
    new_df = processor.transform(ds).to_ddf().compute()
    if cpu:
        assert isinstance(new_df, pd.DataFrame)

    for col, bs in zip(cont_names, boundaries):
        assert np.all(new_df[col].values >= 0)
        assert np.all(new_df[col].values <= len(bs))
        # TODO: add checks for correctness here that don't just
        # repeat the existing logic


@pytest.mark.skipif(not _HAS_GPU, reason="This unittest requires cudf/dask_cudf to run")
@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("cpu", _CPU)
def test_data_stats(tmpdir, df, datasets, engine, cpu):
    # cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y"]
    label_name = ["label"]
    all_cols = cat_names + cont_names + label_name

    dataset = nvtabular.Dataset(df, engine=engine, cpu=cpu)

    data_stats = ops.DataStats()

    features = all_cols >> data_stats
    workflow = nvtabular.Workflow(features)
    workflow.fit(dataset)

    # get the output from the data_stats op
    output = data_stats.output

    # Check Output
    ddf = dask_cudf.from_cudf(df, 2)
    ddf_dtypes = ddf.head(1)
    for col in all_cols:
        # Check dtype
        dtype = ddf_dtypes[col].dtype
        assert output[col]["dtype"] == str(dtype)

        # Identify column type
        if np.issubdtype(dtype, np.floating):
            col_type = "cont"
        else:
            col_type = "cat"

        # Get cardinality for cats
        if col_type == "cat":
            assert output[col]["cardinality"] == ddf[col].nunique().compute()

        # if string, replace string for their lengths for the rest of the computations
        if dtype == "object":
            ddf[col] = ddf[col].map_partitions(lambda x: x.str.len(), meta=("x", int))
            ddf[col].compute()
        # Add list support when cudf supports it:
        # https://github.com/rapidsai/cudf/issues/7157
        # elif col_type == "cat_mh":
        #    ddf[col] = ddf[col].map_partitions(lambda x: x.list.len())

        # Get min,max, and mean
        assert output[col]["min"] == pytest.approx(ddf[col].min().compute())
        assert output[col]["max"] == pytest.approx(ddf[col].max().compute())
        assert output[col]["mean"] == pytest.approx(ddf[col].mean().compute())

        # Get std only for conts
        if col_type == "cont":
            assert output[col]["std"] == pytest.approx(ddf[col].std().compute())

        # Get Percentage of NaNs for all
        assert output[col]["per_nan"] == pytest.approx(
            100 * (1 - ddf[col].count().compute() / len(ddf[col]))
        )


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


@pytest.mark.parametrize("cpu", _CPU)
def test_list_slice(cpu):
    DataFrame = pd.DataFrame if cpu else cudf.DataFrame

    df = DataFrame({"y": [[0, 1, 2, 2, 767], [1, 2, 2, 3], [1, 223, 4]]})

    op = ops.ListSlice(0, 2)
    selector = ColumnSelector(["y"])
    transformed = op.transform(selector, df)
    expected = DataFrame({"y": [[0, 1], [1, 2], [1, 223]]})
    assert_eq(transformed, expected)

    op = ops.ListSlice(3, 5)
    transformed = op.transform(selector, df)
    expected = DataFrame({"y": [[2, 767], [3], []]})
    assert_eq(transformed, expected)

    op = ops.ListSlice(4, 10)
    transformed = op.transform(selector, df)
    expected = DataFrame({"y": [[767], [], []]})
    assert_eq(transformed, expected)

    op = ops.ListSlice(100, 20000)
    transformed = op.transform(selector, df)
    expected = DataFrame({"y": [[], [], []]})
    assert_eq(transformed, expected)

    op = ops.ListSlice(-4)
    transformed = op.transform(selector, df)
    expected = DataFrame({"y": [[1, 2, 2, 767], [1, 2, 2, 3], [1, 223, 4]]})
    assert_eq(transformed, expected)

    op = ops.ListSlice(-3, -1)
    transformed = op.transform(selector, df)
    expected = DataFrame({"y": [[2, 2], [2, 2], [1, 223]]})
    assert_eq(transformed, expected)


@pytest.mark.parametrize("cpu", _CPU)
def test_list_slice_pad(cpu):
    DataFrame = pd.DataFrame if cpu else cudf.DataFrame
    df = DataFrame({"y": [[0, 1, 2, 2, 767], [1, 2, 2, 3], [1, 223, 4]]})

    # 0 pad to 5 elements
    op = ops.ListSlice(5, pad=True)
    selector = ColumnSelector(["y"])
    transformed = op.transform(selector, df)
    expected = DataFrame({"y": [[0, 1, 2, 2, 767], [1, 2, 2, 3, 0], [1, 223, 4, 0, 0]]})
    assert_eq(transformed, expected)

    # make sure we can also pad when start != 0, and when pad_value is set
    op = ops.ListSlice(1, 6, pad=True, pad_value=123)
    selector = ColumnSelector(["y"])
    transformed = op.transform(selector, df)
    expected = DataFrame({"y": [[1, 2, 2, 767, 123], [2, 2, 3, 123, 123], [223, 4, 123, 123, 123]]})
    assert_eq(transformed, expected)

    # we should be able to do pad out negative offsets as well
    op = ops.ListSlice(-4, pad=True, pad_value=-1)
    selector = ColumnSelector(["y"])
    transformed = op.transform(selector, df)
    expected = DataFrame({"y": [[1, 2, 2, 767], [1, 2, 2, 3], [1, 223, 4, -1]]})
    assert_eq(transformed, expected)

    op = ops.ListSlice(-4, -1, pad=True, pad_value=-1)
    selector = ColumnSelector(["y"])
    transformed = op.transform(selector, df)
    expected = DataFrame({"y": [[1, 2, 2], [1, 2, 2], [1, 223, -1]]})
    assert_eq(transformed, expected)

    op = ops.ListSlice(-4, pad=True, pad_value=-1)
    selector = ColumnSelector(["y"])
    transformed = op.transform(selector, df)
    expected = DataFrame({"y": [[1, 2, 2, 767], [1, 2, 2, 3], [1, 223, 4, -1]]})
    assert_eq(transformed, expected)


@pytest.mark.parametrize("cpu", _CPU)
def test_rename(cpu):
    DataFrame = pd.DataFrame if cpu else cudf.DataFrame
    df = DataFrame({"x": [1, 2, 3, 4, 5], "y": [6, 7, 8, 9, 10]})

    selector = ColumnSelector(["x", "y"])

    op = ops.Rename(f=lambda name: name.upper())
    transformed = op.transform(selector, df)
    expected = DataFrame({"X": [1, 2, 3, 4, 5], "Y": [6, 7, 8, 9, 10]})
    assert_eq(transformed, expected)

    op = ops.Rename(postfix="_lower")
    transformed = op.transform(selector, df)
    expected = DataFrame({"x_lower": [1, 2, 3, 4, 5], "y_lower": [6, 7, 8, 9, 10]})
    assert_eq(transformed, expected)

    selector = ColumnSelector(["x"])

    op = ops.Rename(name="z")
    transformed = op.transform(selector, df)
    expected = DataFrame({"z": [1, 2, 3, 4, 5]})
    assert_eq(transformed, expected)

    op = nvt.ops.Rename(f=lambda name: name.upper())
    transformed = op.transform(selector, df)
    expected = DataFrame({"X": [1, 2, 3, 4, 5]})
    assert_eq(transformed, expected)
