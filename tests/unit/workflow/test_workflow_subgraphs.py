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

import os

import numpy as np
import pytest
from pandas.api.types import is_integer_dtype

from merlin.core.utils import set_dask_client
from merlin.dag.ops.subgraph import Subgraph
from nvtabular import Workflow, ops
from tests.conftest import assert_eq


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

    concat_ops = "_FillMissing_1_LogOp_1"
    if replace:
        concat_ops = ""

    df_pp = workflow.transform(dataset).to_ddf().compute()

    if engine == "parquet":
        assert is_integer_dtype(df_pp["name-cat"].dtype)
    assert is_integer_dtype(df_pp["name-string"].dtype)

    subgraph_cat = workflow.get_subworkflow("cat_graph")
    subgraph_cont = workflow.get_subworkflow("cont_graph")
    assert isinstance(subgraph_cat, Workflow)
    assert isinstance(subgraph_cont, Workflow)
    # will not be the same nodes of saved out and loaded back
    if not dump:
        assert subgraph_cat.output_node == cat_features
        assert subgraph_cont.output_node == cont_features
    # check failure path works as expected
    with pytest.raises(ValueError) as exc:
        workflow.get_subworkflow("not_exist")
    assert "No subgraph named" in str(exc.value)

    # test transform results from subgraph
    sub_cat_df = subgraph_cat.transform(dataset).to_ddf().compute()
    assert_eq(sub_cat_df, df_pp[cat_names])

    cont_names = [name + concat_ops for name in cont_names]
    sub_cont_df = subgraph_cont.transform(dataset).to_ddf().compute()
    assert_eq(sub_cont_df[cont_names], df_pp[cont_names])
