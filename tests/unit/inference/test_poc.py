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
import time

import numpy as np
import tensorflow as tf
from feast import FeatureStore

import nvtabular as nvt
from nvtabular import ColumnSchema, Schema
from nvtabular.inference.graph.ensemble import Ensemble
from nvtabular.inference.graph.ops.faiss import QueryFaiss, setup_faiss
from nvtabular.inference.graph.ops.feast import QueryFeast
from nvtabular.inference.graph.ops.session_filter import FilterCandidates
from nvtabular.inference.graph.ops.softmax_sampling import SoftmaxSampling
from nvtabular.inference.graph.ops.tensorflow import PredictTensorflow
from nvtabular.inference.graph.ops.unroll_features import UnrollFeatures
from tests.unit.inference.inference_utils import _run_ensemble_on_tritonserver

# Clean up work:
# - Pull features and mh_features from the schema
# - Clean up input columns that are also params into dependencies
# - Add validation to operators that only require a single input column
# (like SoftmaxSampling and FilterCandidates)
# - Migrate the operators to use cupy if present


def test_poc_ensemble(tmpdir):
    timings = {}
    start = time.time()

    request_schema = Schema(
        [
            ColumnSchema("user_id", dtype=np.int32),
        ]
    )

    def sampled_softmax_loss(y_true, y_pred):
        return tf.nn.sampled_softmax_loss(
            weights=item_embeddings,
            biases=tf.fill((item_embeddings.shape[0],), 0.01),
            labels=y_true,
            inputs=y_pred,
            num_sampled=20,
            num_classes=item_embeddings.shape[0],
        )

    feast_repo_path = "/nvtabular/examples/end-to-end-poc/feature_repo"

    retrieval_model_path = (
        "/nvtabular/examples/end-to-end-poc/models/movielens_retrieval_tf/1/model.savedmodel/"
    )
    retrieval_model = tf.keras.models.load_model(
        retrieval_model_path, custom_objects={"sampled_softmax_loss": sampled_softmax_loss}
    )
    item_embeddings = retrieval_model.input_layer.embedding_tables["movie_ids"].numpy()

    ranking_model_path = (
        "/nvtabular/examples/end-to-end-poc/models/movielens_ranking_tf/1/model.savedmodel/"
    )

    faiss_index_path = tmpdir + "/index.faiss"
    setup_faiss(item_embeddings, str(faiss_index_path))

    timings["build_steps_b4_ensemble"] = time.time() - start
    start = time.time()

    # TODO: Make it possible to pass multiple user ids in the request

    feature_store = FeatureStore(feast_repo_path)

    user_features = ["user_id"] >> QueryFeast.from_feature_view(
        store=feature_store, path=feast_repo_path, view="user_features", column="user_id"
    )

    retrieval = (
        user_features
        >> PredictTensorflow(
            retrieval_model_path,
            custom_objects={"sampled_softmax_loss": sampled_softmax_loss},
        )
        # TODO: Should only have a single input column and use that
        # TODO: Replace index path with FAISS Index object
        # TODO: Mock out FAISS
        >> QueryFaiss(faiss_index_path, query_vector_col="output_1", topk=100)
    )

    # TODO: Should only have a single input column and use that
    # TODO: Make the filter_col a dependency instead of a string
    filtering = retrieval["candidate_ids"] + user_features["movie_ids_1"] >> FilterCandidates(
        candidate_col="candidate_ids", filter_col="movie_ids_1"
    )

    # TODO: Mock out FeatureStore for the test (so we don't need actual Feast here)
    item_features = filtering >> QueryFeast.from_feature_view(
        store=feature_store,
        path=feast_repo_path,
        view="movie_features",
        column="filtered_ids",
        output_prefix="movie",
        include_id=True,
    )

    # TODO: Make the user_features a dependency instead of a list of strings
    user_features_to_unroll = [
        "genres_1",
        "genres_2",
        "movie_ids_1",
        "movie_ids_2",
        "search_terms_1",
        "search_terms_2",
        "movie_id_count",
    ]
    combined_features = user_features + item_features >> UnrollFeatures(
        "movie_id", user_features_to_unroll, unrolled_prefix="user"
    )

    ranking = combined_features >> PredictTensorflow(ranking_model_path)

    # TODO: Make the relevance_col a dependency instead of a string
    # TODO: Should only have a single input column and use that (remove "movie_id" param)
    #  ordering = combined_features["movie_id"] +  >> SoftmaxSampling(
    ordering = (combined_features + ranking)["movie_id", "output_1"] >> SoftmaxSampling(
        "movie_id", relevance_col="output_1", topk=10, temperature=20.0
    )

    export_path = str("/nvtabular/test_poc/")

    ensemble = Ensemble(ordering, request_schema)

    timings["ensemble_create"] = time.time() - start
    start = time.time()

    ens_config, node_configs = ensemble.export(export_path)

    timings["export_ensemble"] = time.time() - start

    request = nvt.dispatch._make_df({"user_id": [1]})
    request["user_id"] = request["user_id"].astype(np.int32)

    start = time.time()

    response = _run_ensemble_on_tritonserver(
        export_path, ensemble.graph.output_schema.column_names, request, "ensemble_model"
    )

    timings["run_triton"] = time.time() - start

    assert response is not None
    assert len(response.as_numpy("ordered_ids")) == 10
    # breakpoint()
