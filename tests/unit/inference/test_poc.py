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
import numpy as np

import nvtabular as nvt
from nvtabular import ColumnSchema, Schema
from nvtabular.inference.graph.ops.feast import QueryFeast
from nvtabular.inference.graph.ops.milvus import QueryMilvus
from nvtabular.inference.graph.ops.tensorflow import PredictTensorflow
from nvtabular.inference.graph.ops.unroll_features import UnrollFeatures
from nvtabular.inference.graph.ops.session_filter import FilterCandidates
from nvtabular.inference.graph.ops.softmax_sampling import SoftMaxSampling
from nvtabular.inference.graph.ensemble import Ensemble

from tests.unit.inference.inference_utils import _run_ensemble_on_tritonserver, create_tf_model

import tensorflow as tf
import pymilvus


def test_poc(tmpdir):

    # TODO: Run all the notebooks, save models and workflows outside the container

    # Triton Set-up
    request_schema = Schema([ColumnSchema("user_id", dtype=np.int32)])

    # NVT Set-up
    # retrieval_workflow = nvt.load("some_workflow_path")
    # ranking_workflow = nvt.load("some_other_path")
    # workflow = nvt.load("the_whole_thing_path")

    # Feast Set-up
    feast_repo_path = "/feast"

    # Tensorflow Set-up
    retrieval_model = create_tf_model(user_cat_columns, user_mh_columns, embedding_table_shapes)
    ranking_model = create_tf_model(cat_columns, mh_columns, embedding_shapes)

    # Retrieval return type: list of floats (i.e. user embedding)
    # Ranking return type: list of floats (i.e. item scores)

    # Milvus Set-up
    dim = 128
    default_fields = [
        pymilvus.FieldSchema(name="item_id", dtype=pymilvus.DataType.INT64, is_primary=True),
        pymilvus.FieldSchema(name="item_vector", dtype=pymilvus.DataType.FLOAT_VECTOR, dim=dim),
    ]
    default_schema = pymilvus.CollectionSchema(
        fields=default_fields, description="MovieLens item vectors"
    )
    milvus_collection = pymilvus.Collection(
        name="movielens_retrieval_tf", data=None, schema=default_schema
    )

    index_name = "item_vectors"

    # Retrieval
    retrieval = (
        QueryFeast(
            feast_repo_path,
            "user_id",
            "user_features",
            ["movie_id_count"],
            ["movie_ids", "genres", "search_terms"],
        )
        # >> TransformWorkflow(retrieval_workflow)
        >> PredictTensorflow(retrieval_model)
        >> QueryMilvus("milvus-standalone", 19530, milvus_collection)
        >> QueryFeast(
            feast_repo_path,
            "movie_id",
            "movie_features",
            ["tags_nunique"],
            ["genres", "tags_unique"],
        )
    )

    # Filtering
    filtering = retrieval >> FilterCandidates("candidate_movie_ids", "session_movie_ids")

    # Ranking
    user_features = [
        "candidate_movie_ids",
        "movie_id_count",
        "movie_ids__values",
        "movie_ids__nnzs",
        "genres__values",
        "genres__nnzs",
        "search_terms__values",
        "search_terms__nnzs",
    ]

    ranking = (
        filtering
        >> UnrollFeatures("candidate_movie_ids", user_features)
        # >> TransformWorkflow(ranking_workflow)
        >> PredictTensorflow(ranking_model)
    )

    # Ordering
    ordering = ranking >> SoftMaxSampling("candidate_movie_ids", "predicted_scores")

    export_path = str(tmpdir)

    ensemble = Ensemble(ordering)
    ensemble.export(export_path, request_schema)

    request_user_ids = nvt.dispatch._make_df({"user_ids": [1, 2, 3, 4]})

    response = _run_ensemble_on_tritonserver(export_path, ["output"], request_user_ids, "test_name")

    assert response is not None
