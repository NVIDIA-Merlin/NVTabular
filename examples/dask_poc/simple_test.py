import os
import glob
import shutil
import subprocess as sp

import dask_cudf
import cudf
import cupy
from dask.datasets import timeseries
from dask.dataframe import assert_eq


test_dataset_path = "/datasets/rzamora/timeseries.pq"
out_dir = "/datasets/rzamora/debug"
dask_workspace = "/raid/dask_space/rzamora"
devices = "2,3"
protocol = "tcp"
row_group_chunk = "4"
nsplits = "2"
freq_limit = "0"

if os.path.isdir(out_dir):
    shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    
if True:
    # Remove old dataset
    if os.path.isdir(test_dataset_path):
        shutil.rmtree(test_dataset_path)

    # Write test dataset
    ddf_d = timeseries(
        start='2000-01-01',
        end='2001-01-01',
        freq='1h',
        partition_freq='30d',
        dtypes={'x': float, 'y': float, 'name': str, 'id1': int, 'id2': int},
        id_lam=1_000_000_000,
    ).reset_index(drop=True)
    ddf = dask_cudf.from_dask_dataframe(ddf_d)
    del ddf_d

    def _add_nulls(df):
        df["x"][df["x"] < 0.0] = None
        df["id1"][df["id1"] < 1000] = None
        df["name"][df["name"] == "Edith"] = None
        return df
    ddf = ddf.map_partitions(_add_nulls, meta=ddf._meta)

    ddf.to_parquet(
        test_dataset_path,
        write_index=False,
        write_metadata_file=True,
    )
    del ddf


# Run Benchmark
cmd = [
    "python",
    "./criteo_preprocess.py",
    "-d", devices,
    "--data-path", test_dataset_path,
    "--out-path", out_dir,
    "--dask-workspace", dask_workspace,
    "-p", protocol,
    "-r", row_group_chunk,
    "-s", nsplits,
    "-f", freq_limit,
    "--cat-names", "id1,id2,name",
    "--cat-cache", "False,False,True",
    "--cat-splits", "2,2,1",
    "--cont-names", "x,y",
    "--no-rmm-pool",
    "--worker-shuffle",
]
print("\n" + " ".join(cmd) + "\n")
sp.run(cmd)


# Read back dataset
result_paths = glob.glob("/".join([out_dir,"processed","*.parquet"]))
result = dask_cudf.read_parquet(result_paths, index=False).compute()

# Read original dataset
df0 = dask_cudf.read_parquet(test_dataset_path, index=False).compute()


# Simple checks
assert len(df0) == len(result)
assert result["x"].min() == 0.0
assert result["y"].min() == 0.0
assert result["x"].isna().sum() == 0
assert result["y"].isna().sum() == 0

# Check category counts
if freq_limit == "0":
    assert len(df0["id1"].unique()) == len(result["id1"].unique())
    assert len(df0["id2"].unique()) == len(result["id2"].unique())
    assert len(df0["name"].unique()) == len(result["name"].unique())

    df0["_count"] = cupy.ones(len(df0))
    result["_count"] = cupy.ones(len(result))
    for col in ["id1", "id2", "name"]:

        expect = df0.groupby(
            col, dropna=False
        ).count()["_count"].sort_values("_count")

        got = result.groupby(
            col, dropna=False
        ).count()["_count"].sort_values("_count")

        assert_eq(expect, got, check_index=False)


print("Done Successfully.")
