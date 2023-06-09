#
# Copyright (c) 2023, NVIDIA CORPORATION.
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

import numpy as np
import pytest
from pandas.api.types import is_integer_dtype

import nvtabular as nvt
from merlin.core import dispatch
from merlin.core.dispatch import HAS_GPU
from merlin.core.utils import set_dask_client
from merlin.dag.ops.subgraph import Subgraph
from nvtabular import Dataset, Workflow, ops
from tests.conftest import get_cats


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("dump", [True, False])
@pytest.mark.parametrize("replace", [True, False])
def test_workflow_subgraphs(tmpdir, client, df, dataset, gpu_memory_frac, engine, dump, replace):
    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    norms = ops.Normalize()
    cat_features = cat_names >> ops.Categorify()
    if replace:
        cont_features = cont_names >> ops.FillMissing() >> ops.LogOp >> norms
    else:
        fillmissing_logop = (
            cont_names
            >> ops.FillMissing()
            >> ops.LogOp
            >> ops.Rename(postfix="_FillMissing_1_LogOp_1")
        )
        cont_features = cont_names + fillmissing_logop >> norms

    set_dask_client(client=client)
    wkflow_ops = Subgraph("cat_graph", cat_features) + Subgraph("cont_graph", cont_features)
    workflow = Workflow(wkflow_ops + label_name)

    workflow.fit(dataset)

    if dump:
        workflow_dir = os.path.join(tmpdir, "workflow")
        workflow.save(workflow_dir)
        workflow = None

        workflow = Workflow.load(workflow_dir)

    def get_norms(tar):
        ser_median = tar.dropna().quantile(0.5, interpolation="linear")
        gdf = tar.fillna(ser_median)
        gdf = np.log(gdf + 1)
        return gdf

    # Check mean and std - No good right now we have to add all other changes; Clip, Log

    concat_ops = "_FillMissing_1_LogOp_1"
    if replace:
        concat_ops = ""
    assert math.isclose(get_norms(df.x).mean(), norms.means["x" + concat_ops], rel_tol=1e-1)
    assert math.isclose(get_norms(df.y).mean(), norms.means["y" + concat_ops], rel_tol=1e-1)

    assert math.isclose(get_norms(df.x).std(), norms.stds["x" + concat_ops], rel_tol=1e-1)
    assert math.isclose(get_norms(df.y).std(), norms.stds["y" + concat_ops], rel_tol=1e-1)
    # Check that categories match
    if engine == "parquet":
        cats_expected0 = df["name-cat"].unique().values_host if HAS_GPU else df["name-cat"].unique()
        cats0 = get_cats(workflow, "name-cat")
        # adding the None entry as a string because of move from gpu
        assert all(cat in sorted(cats_expected0.tolist()) for cat in cats0.tolist())
        assert len(cats0.tolist()) == len(cats_expected0.tolist())
    cats_expected1 = (
        df["name-string"].unique().values_host if HAS_GPU else df["name-string"].unique()
    )
    cats1 = get_cats(workflow, "name-string")
    # adding the None entry as a string because of move from gpu
    assert all(cat in sorted(cats_expected1.tolist()) for cat in cats1.tolist())
    assert len(cats1.tolist()) == len(cats_expected1.tolist())

    # Write to new "shuffled" and "processed" dataset
    workflow.transform(dataset).to_parquet(
        tmpdir,
        out_files_per_proc=10,
        shuffle=nvt.io.Shuffle.PER_PARTITION,
    )

    dataset_2 = Dataset(glob.glob(str(tmpdir) + "/*.parquet"), part_mem_fraction=gpu_memory_frac)

    df_pp = dispatch.concat(list(dataset_2.to_iter()), axis=0)

    if engine == "parquet":
        assert is_integer_dtype(df_pp["name-cat"].dtype)
    assert is_integer_dtype(df_pp["name-string"].dtype)

    num_rows, num_row_groups, col_names = dispatch.read_parquet_metadata(str(tmpdir) + "/_metadata")
    assert num_rows == len(df_pp)

    subgraph_cat = workflow.get_subgraph("cat_graph")
    subgraph_cont = workflow.get_subgraph("cont_graph")
    assert isinstance(subgraph_cat, Workflow)
    assert isinstance(subgraph_cont, Workflow)
    # will not be the same nodes of saved out and loaded back
    if not dump:
        assert subgraph_cat.output_node == cat_features
        assert subgraph_cont.output_node == cont_features
    # check failure path works as expected
    with pytest.raises(ValueError) as exc:
        workflow.get_subgraph("not_exist")
    assert "No subgraph named" in str(exc.value)
