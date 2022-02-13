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
import tensorflow as tf

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


def test_poc_ensemble(tmpdir):
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
    feast_user_in_schema = Schema([ColumnSchema("user_id", dtype=np.int32)])

    # TODO: Generate this from the Feast schema
    feast_user_out_schema = Schema(
        [
            ColumnSchema("movie_id_count", dtype=np.int32, _is_list=False),
            ColumnSchema("movie_ids_1", dtype=np.int32, _is_list=True),
            ColumnSchema("movie_ids_2", dtype=np.int64, _is_list=True),
            ColumnSchema("search_terms_1", dtype=np.int32, _is_list=True),
            ColumnSchema("search_terms_2", dtype=np.int64, _is_list=True),
            ColumnSchema("genres_1", dtype=np.int32, _is_list=True),
            ColumnSchema("genres_2", dtype=np.int64, _is_list=True),
        ]
    )

    # TODO: Generate this from the Feast schema
    feast_item_out_schema = Schema(
        [
            ColumnSchema("movie_id", dtype=np.int32),
            ColumnSchema("tags_nunique", dtype=np.int32, _is_list=False),
            ColumnSchema("tags_unique_3", dtype=np.int32, _is_list=True, _is_ragged=True),
            ColumnSchema("tags_unique_4", dtype=np.int64, _is_list=True),
            ColumnSchema("genres_3", dtype=np.int32, _is_list=True, _is_ragged=True),
            ColumnSchema("genres_4", dtype=np.int64, _is_list=True),
        ]
    )

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

    user_features = ["user_id"] >> QueryFeast(
        feast_repo_path,
        entity_view="user_features",
        entity_id="user_id",
        entity_column="user_id",
        features=["movie_id_count"],
        mh_features=["movie_ids", "genres", "search_terms"],
        input_schema=feast_user_in_schema,
        output_schema=feast_user_out_schema,
    )

    retrieval = (
        user_features
        >> PredictTensorflow(
            retrieval_model_path, custom_objects={"sampled_softmax_loss": sampled_softmax_loss}
        )
        >> QueryFaiss(faiss_index_path, query_vector_col="output_1", topk=100)
    )

    filtering = user_features["movie_ids_1"] + retrieval["candidate_ids"] >> FilterCandidates(
        "candidate_ids", "movie_ids_1"
    )

    item_features = filtering >> QueryFeast(
        feast_repo_path,
        entity_view="movie_features",
        entity_id="movie_id",
        entity_column="filtered_ids",
        features=["tags_nunique"],
        mh_features=["genres", "tags_unique"],
        input_schema=Schema([ColumnSchema("filtered_ids", dtype=np.int32)]),
        output_schema=feast_item_out_schema,
        suffix_int=3,
    )

    ranking = (
        user_features + item_features
        >> UnrollFeatures("movie_id", feast_user_out_schema.column_names)
        # TODO: Figure out how to align column names with what the second model is expecting
        >> PredictTensorflow(ranking_model_path)
    )

    ordering = ranking >> SoftmaxSampling(
        "movie_id", relevance_col="output_1", topk=10, temperature=20.0
    )

    export_path = str(tmpdir)

    ensemble = Ensemble(ordering, request_schema)
    ens_config, node_configs = ensemble.export(export_path)

    request = nvt.dispatch._make_df({"user_id": [1]})
    request["user_id"] = request["user_id"].astype(np.int32)

    response = _run_ensemble_on_tritonserver(
        export_path, ensemble.graph.output_schema.column_names, request, "ensemble_model"
    )

    assert response is not None
