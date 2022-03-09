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
from distutils.spawn import find_executable

import numpy as np
import pytest

from merlin.core.dispatch import make_df
from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.dag.ops.session_filter import FilterCandidates
from merlin.systems.dag.ops.softmax_sampling import SoftmaxSampling
from tests.unit.systems.inference_utils import _run_ensemble_on_tritonserver  # noqa

TRITON_SERVER_PATH = find_executable("tritonserver")


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
def test_softmax_sampling(tmpdir):
    request_schema = Schema(
        [
            ColumnSchema("movie_ids", dtype=np.int32),
            ColumnSchema("output_1", dtype=np.float32),
        ]
    )

    combined_features = {
        "movie_ids": np.random.randint(0, 10000, 100).astype(np.int32),
        "output_1": np.random.random(100).astype(np.float32),
    }

    request = make_df(combined_features)

    ordering = ["movie_ids"] >> SoftmaxSampling(relevance_col="output_1", topk=10, temperature=20.0)

    ensemble = Ensemble(ordering, request_schema)
    ens_config, node_configs = ensemble.export(tmpdir)

    response = _run_ensemble_on_tritonserver(
        tmpdir, ensemble.graph.output_schema.column_names, request, "ensemble_model"
    )
    assert response is not None
    assert len(response.as_numpy("ordered_ids")) == 10


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
def test_filter_candidates(tmpdir):
    request_schema = Schema(
        [
            ColumnSchema("candidate_ids", dtype=np.int32),
            ColumnSchema("movie_ids", dtype=np.int32),
        ]
    )

    candidate_ids = np.random.randint(1, 100000, 100).astype(np.int32)
    movie_ids_1 = np.zeros(100, dtype=np.int32)
    movie_ids_1[:20] = np.unique(candidate_ids)[:20]

    combined_features = {
        "candidate_ids": candidate_ids,
        "movie_ids": movie_ids_1,
    }

    request = make_df(combined_features)

    filtering = ["candidate_ids"] >> FilterCandidates(filter_out=["movie_ids"])

    ensemble = Ensemble(filtering, request_schema)
    ens_config, node_configs = ensemble.export(tmpdir)

    response = _run_ensemble_on_tritonserver(
        tmpdir, ensemble.graph.output_schema.column_names, request, "ensemble_model"
    )
    assert response is not None
    assert len(response.as_numpy("filtered_ids")) == 80
