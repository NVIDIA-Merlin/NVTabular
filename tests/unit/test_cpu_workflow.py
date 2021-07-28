import glob
import math
import os

import pandas as pd
import pyarrow.parquet as pq
import pytest
from pandas.api.types import is_integer_dtype

import nvtabular as nvt
from nvtabular import Dataset, Workflow, ops
from tests.conftest import get_cats

try:
    import cudf
except ImportError:
    cudf = None


@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("dump", [True, False])
@pytest.mark.parametrize("cpu", [True])
def test_cpu_workflow(tmpdir, df, dataset, cpu, engine, dump):
    # Make sure we are in cpu formats
    if cudf and isinstance(df, cudf.DataFrame):
        df = df.to_pandas()

    if cpu:
        dataset.to_cpu()

    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    norms = ops.Normalize()
    conts = cont_names >> ops.FillMissing() >> ops.Clip(min_value=0) >> norms
    cats = cat_names >> ops.Categorify()
    workflow = nvt.Workflow(conts + cats + label_name)

    workflow.fit(dataset)
    if dump:
        workflow_dir = os.path.join(tmpdir, "workflow")
        workflow.save(workflow_dir)
        workflow = None

        workflow = Workflow.load(workflow_dir)

    def get_norms(tar: pd.Series):
        df = tar.fillna(0)
        df = df * (df >= 0).astype("int")
        return df

    assert math.isclose(get_norms(df.x).mean(), norms.means["x"], rel_tol=1e-4)
    assert math.isclose(get_norms(df.y).mean(), norms.means["y"], rel_tol=1e-4)
    assert math.isclose(get_norms(df.x).std(), norms.stds["x"], rel_tol=1e-3)
    assert math.isclose(get_norms(df.y).std(), norms.stds["y"], rel_tol=1e-3)

    # Check that categories match
    if engine == "parquet":
        cats_expected0 = df["name-cat"].unique()
        cats0 = get_cats(workflow, "name-cat", cpu=True)
        # adding the None entry as a string because of move from gpu
        assert all(cat in [None] + sorted(cats_expected0.tolist()) for cat in cats0.tolist())
        assert len(cats0.tolist()) == len(cats_expected0.tolist() + [None])
    cats_expected1 = df["name-string"].unique()
    cats1 = get_cats(workflow, "name-string", cpu=True)
    # adding the None entry as a string because of move from gpu
    assert all(cat in [None] + sorted(cats_expected1.tolist()) for cat in cats1.tolist())
    assert len(cats1.tolist()) == len(cats_expected1.tolist() + [None])

    # Write to new "shuffled" and "processed" dataset
    workflow.transform(dataset).to_parquet(
        output_path=tmpdir, out_files_per_proc=10, shuffle=nvt.io.Shuffle.PER_PARTITION
    )

    dataset_2 = Dataset(glob.glob(str(tmpdir) + "/*.parquet"), cpu=cpu)

    df_pp = pd.concat(list(dataset_2.to_iter()), axis=0)

    if engine == "parquet":
        assert is_integer_dtype(df_pp["name-cat"].dtype)
    assert is_integer_dtype(df_pp["name-string"].dtype)

    metadata = pq.read_metadata(str(tmpdir) + "/_metadata")
    assert metadata.num_rows == len(df_pp)
