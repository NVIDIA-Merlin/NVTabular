#
# Copyright (c) 2020, NVIDIA CORPORATION.
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
import math
import os

import cudf
import dask_cudf
import numpy as np
import pytest
from dask.dataframe import assert_eq

from nvtabular import Dataset, Workflow
from nvtabular import ops as ops
from nvtabular.io.shuffle import Shuffle
from tests.conftest import allcols_csv, mycols_csv, mycols_pq


# Dummy operator logic to test stats
# TODO: Possibly add public API to add
#       standalone Stat Ops
def _dummy_op_logic(gdf, target_columns, _id="dummy", **kwargs):
    cont_names = target_columns
    if not cont_names:
        return gdf
    new_gdf = gdf[cont_names]
    new_cols = [f"{col}_{_id}" for col in new_gdf.columns]
    new_gdf.columns = new_cols
    return new_gdf


@pytest.mark.parametrize("part_mem_fraction", [0.01])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("freq_threshold", [0, 150])
@pytest.mark.parametrize("cat_cache", ["device", None])
@pytest.mark.parametrize("on_host", [True, False])
@pytest.mark.parametrize("shuffle", [Shuffle.PER_WORKER, None])
def test_dask_workflow_api_dlrm(
    client, tmpdir, datasets, freq_threshold, part_mem_fraction, engine, cat_cache, on_host, shuffle
):
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])
    if engine == "parquet":
        df1 = cudf.read_parquet(paths[0])[mycols_pq]
        df2 = cudf.read_parquet(paths[1])[mycols_pq]
    elif engine == "csv":
        df1 = cudf.read_csv(paths[0], header=0)[mycols_csv]
        df2 = cudf.read_csv(paths[1], header=0)[mycols_csv]
    else:
        df1 = cudf.read_csv(paths[0], names=allcols_csv)[mycols_csv]
        df2 = cudf.read_csv(paths[1], names=allcols_csv)[mycols_csv]
    df0 = cudf.concat([df1, df2], axis=0)

    if engine == "parquet":
        cat_names = ["name-cat", "name-string"]
    else:
        cat_names = ["name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    processor = Workflow(
        client=client, cat_names=cat_names, cont_names=cont_names, label_name=label_name
    )

    processor.add_feature([ops.FillMissing(), ops.Clip(min_value=0), ops.LogOp()])
    processor.add_preprocess(
        ops.Categorify(
            freq_threshold=freq_threshold,
            out_path=str(tmpdir),
            cat_cache=cat_cache,
            on_host=on_host,
        )
    )
    processor.finalize()

    if engine in ("parquet", "csv"):
        dataset = Dataset(paths, part_mem_fraction=part_mem_fraction)
    else:
        dataset = Dataset(paths, names=allcols_csv, part_mem_fraction=part_mem_fraction)
    output_path = os.path.join(tmpdir, "processed")
    processor.apply(dataset, output_path=output_path, shuffle=shuffle)

    # Can still access the final ddf if we didn't shuffle
    if not shuffle:
        result = processor.get_ddf().compute()
        assert len(df0) == len(result)
        assert result["x"].min() == 0.0
        assert result["x"].isna().sum() == 0
        assert result["y"].min() == 0.0
        assert result["y"].isna().sum() == 0

        # Check category counts
        cat_expect = df0.groupby("name-string").agg({"name-string": "count"}).reset_index(drop=True)
        cat_result = (
            result.groupby("name-string").agg({"name-string": "count"}).reset_index(drop=True)
        )
        if freq_threshold:
            cat_expect = cat_expect[cat_expect["name-string"] >= freq_threshold]
            # Note that we may need to skip the 0th element in result (null mapping)
            assert_eq(
                cat_expect,
                cat_result.iloc[1:] if len(cat_result) > len(cat_expect) else cat_result,
                check_index=False,
            )
        else:
            assert_eq(cat_expect, cat_result)

        # Read back from disk
        df_disk = dask_cudf.read_parquet(output_path, index=False).compute()
        for col in df_disk:
            assert_eq(result[col], df_disk[col])

    else:
        # Read back from disk
        df_disk = dask_cudf.read_parquet(output_path, index=False).compute()
        assert len(df0) == len(df_disk)


@pytest.mark.parametrize("part_mem_fraction", [0.01])
def test_dask_groupby_stats(client, tmpdir, datasets, part_mem_fraction):

    engine = "parquet"
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])
    df1 = cudf.read_parquet(paths[0])[mycols_pq]
    df2 = cudf.read_parquet(paths[1])[mycols_pq]
    df0 = cudf.concat([df1, df2], axis=0)

    cat_names = ["name-cat", "name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    processor = Workflow(
        client=client, cat_names=cat_names, cont_names=cont_names, label_name=label_name
    )

    processor.add_preprocess(
        ops.JoinGroupby(
            cont_names=cont_names, stats=["count", "sum", "std", "min"], out_path=str(tmpdir)
        )
    )
    processor.finalize()

    dataset = Dataset(paths, part_mem_fraction=part_mem_fraction)
    processor.apply(dataset)
    result = processor.get_ddf().compute(scheduler="synchronous")

    # Validate result
    assert len(df0) == len(result)
    assert "name-cat_x_std" in result.columns
    assert "name-cat_x_var" not in result.columns
    assert "name-string_x_std" in result.columns
    assert "name-string_x_var" not in result.columns

    # Check "count"
    assert_eq(
        result[["name-cat", "name-cat_count"]]
        .drop_duplicates()
        .sort_values("name-cat")["name-cat_count"],
        df0.groupby("name-cat").agg({"x": "count"})["x"].astype(np.int64),
        check_index=False,
        check_dtype=False,  # May get int64 vs int32
        check_names=False,
    )

    # Check "min"
    assert_eq(
        result[["name-string", "name-string_x_min"]]
        .drop_duplicates()
        .sort_values("name-string")["name-string_x_min"],
        df0.groupby("name-string").agg({"x": "min"})["x"],
        check_index=False,
        check_names=False,
    )

    # Check "std"
    assert_eq(
        result[["name-string", "name-string_x_std"]]
        .drop_duplicates()
        .sort_values("name-string")["name-string_x_std"],
        df0.groupby("name-string").agg({"x": "std"})["x"],
        check_index=False,
        check_names=False,
    )


@pytest.mark.parametrize("part_mem_fraction", [0.01])
@pytest.mark.parametrize("use_client", [True, False])
def test_cats_and_groupby_stats(client, tmpdir, datasets, part_mem_fraction, use_client):

    engine = "parquet"
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])

    cat_names = ["name-cat", "name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    processor = Workflow(
        client=client if use_client else None,
        cat_names=cat_names,
        cont_names=cont_names,
        label_name=label_name,
    )

    processor.add_preprocess(ops.Categorify(out_path=str(tmpdir), freq_threshold=10, on_host=True))

    processor.add_cat_feature(
        ops.JoinGroupby(cont_names=cont_names, stats=["count", "sum"], out_path=str(tmpdir))
    )

    processor.finalize()
    dataset = Dataset(paths, part_mem_fraction=part_mem_fraction)

    processor.apply(dataset, output_path=str(tmpdir))
    result = processor.get_ddf().compute()

    assert "name-cat_x_sum" in result.columns
    assert "name-string_x_sum" in result.columns


@pytest.mark.parametrize("engine", ["parquet"])
def test_dask_minmax_dummyop(client, tmpdir, datasets, engine):

    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])
    cat_names = ["name-cat", "name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    class DummyOp(ops.DFOperator):

        default_in, default_out = "continuous", "continuous"

        @property
        def req_stats(self):
            return [ops.MinMax()]

        def op_logic(self, *args, **kwargs):
            return _dummy_op_logic(*args, _id=self._id, **kwargs)

    processor = Workflow(
        client=client, cat_names=cat_names, cont_names=cont_names, label_name=label_name
    )
    processor.add_preprocess(DummyOp())
    processor.finalize()

    dataset = Dataset(paths, engine)
    processor.apply(dataset)
    result = processor.get_ddf().compute()

    assert math.isclose(result.x.min(), processor.stats["mins"]["x"], rel_tol=1e-3)
    assert math.isclose(result.y.min(), processor.stats["mins"]["y"], rel_tol=1e-3)
    assert math.isclose(result.id.min(), processor.stats["mins"]["id"], rel_tol=1e-3)
    assert math.isclose(result.x.max(), processor.stats["maxs"]["x"], rel_tol=1e-3)
    assert math.isclose(result.y.max(), processor.stats["maxs"]["y"], rel_tol=1e-3)
    assert math.isclose(result.id.max(), processor.stats["maxs"]["id"], rel_tol=1e-3)


@pytest.mark.parametrize("engine", ["parquet"])
def test_dask_median_dummyop(client, tmpdir, datasets, engine):

    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])
    cat_names = ["name-cat", "name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    class DummyOp(ops.DFOperator):

        default_in, default_out = "continuous", "continuous"

        @property
        def req_stats(self):
            return [ops.Median()]

        def op_logic(self, *args, **kwargs):
            return _dummy_op_logic(*args, _id=self._id, **kwargs)

    processor = Workflow(
        client=client, cat_names=cat_names, cont_names=cont_names, label_name=label_name
    )
    processor.add_preprocess(DummyOp())
    processor.finalize()

    dataset = Dataset(paths, engine)
    processor.apply(dataset)
    result = processor.get_ddf().compute()

    # TODO: Improve the accuracy! "tidigest" with crick could help,
    #       but current version seems to have cupy/numpy problems here
    medians = result[cont_names].quantile(q=0.5)
    assert math.isclose(medians["x"], processor.stats["medians"]["x"], abs_tol=1e-1)
    assert math.isclose(medians["y"], processor.stats["medians"]["y"], abs_tol=1e-1)
    assert math.isclose(medians["id"], processor.stats["medians"]["id"], rel_tol=1e-2)


@pytest.mark.parametrize("engine", ["parquet"])
def test_dask_normalize(client, tmpdir, datasets, engine):

    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])
    df1 = cudf.read_parquet(paths[0])[mycols_pq]
    df2 = cudf.read_parquet(paths[1])[mycols_pq]
    df0 = cudf.concat([df1, df2], axis=0)

    cat_names = ["name-cat", "name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    processor = Workflow(
        client=client, cat_names=cat_names, cont_names=cont_names, label_name=label_name
    )
    processor.add_preprocess([ops.FillMissing(), ops.Normalize()])
    processor.finalize()

    dataset = Dataset(paths, engine)
    processor.apply(dataset)
    result = processor.get_ddf().compute()

    # Make sure we collected accurate statistics
    means = df0[cont_names].mean()
    stds = df0[cont_names].std()
    counts = df0[cont_names].count()
    for name in cont_names:
        assert math.isclose(means[name], processor.stats["means"][name], rel_tol=1e-3)
        assert math.isclose(stds[name], processor.stats["stds"][name], rel_tol=1e-3)
        assert math.isclose(counts[name], processor.stats["counts"][name], rel_tol=1e-3)

    # New (normalized) means should all be close to zero
    new_means = result[cont_names].mean()
    for name in cont_names:
        assert new_means[name] < 1e-3
