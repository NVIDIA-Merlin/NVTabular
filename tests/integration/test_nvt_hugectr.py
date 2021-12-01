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

import gc
import glob
import json
import os
import shutil
from os import path

import cudf
import pytest

try:
    import hugectr
    from hugectr.inference import CreateInferenceSession, InferenceParams
    from mpi4py import MPI  # noqa pylint: disable=unused-import
except ImportError:
    hugectr = None

import warnings
from distutils.spawn import find_executable

import numpy as np
import pandas as pd
from common.parsers.benchmark_parsers import create_bench_result
from common.utils import _run_query
from sklearn.model_selection import train_test_split

import nvtabular as nvt
import tests.conftest as test_utils
from nvtabular.inference.triton import export_hugectr_ensemble
from nvtabular.ops import get_embedding_sizes
from nvtabular.utils import download_file

DIR = "/model/"
DATA_DIR = DIR + "data/"
TEMP_DIR = DIR + "temp_hugectr/"
MODEL_DIR = DIR + "models/"
TRAIN_DIR = MODEL_DIR + "test_model/1/"
NETWORK_FILE = TRAIN_DIR + "model.json"
DENSE_FILE = TRAIN_DIR + "_dense_1900.model"
SPARSE_FILES = TRAIN_DIR + "0_sparse_1900.model"
MODEL_NAME = "test_model"

CATEGORICAL_COLUMNS = ["userId", "movieId", "new_cat1"]
LABEL_COLUMNS = ["rating"]
TEST_N_ROWS = 64

TRITON_SERVER_PATH = find_executable("tritonserver")
TRITON_DEVICE_ID = "1"


def test_training():
    # Download & Convert data
    download_file(
        "http://files.grouplens.org/datasets/movielens/ml-25m.zip",
        os.path.join(DATA_DIR, "ml-25m.zip"),
    )

    ratings = cudf.read_csv(os.path.join(DATA_DIR, "ml-25m", "ratings.csv"))
    ratings["new_cat1"] = ratings["userId"] / ratings["movieId"]
    ratings["new_cat1"] = ratings["new_cat1"].astype("int64")
    ratings.head()

    ratings = ratings.drop("timestamp", axis=1)
    train, valid = train_test_split(ratings, test_size=0.2, random_state=42)

    train.to_parquet(DATA_DIR + "train.parquet")
    valid.to_parquet(DATA_DIR + "valid.parquet")

    del train
    del valid
    gc.collect()

    # Perform ETL with NVTabular
    cat_features = CATEGORICAL_COLUMNS >> nvt.ops.Categorify(cat_cache="device")
    ratings = nvt.ColumnSelector(["rating"]) >> nvt.ops.LambdaOp(
        lambda col: (col > 3).astype("int8")
    )
    output = cat_features + ratings

    workflow = nvt.Workflow(output)

    train_dataset = nvt.Dataset(DATA_DIR + "train.parquet", part_size="100MB")
    valid_dataset = nvt.Dataset(DATA_DIR + "valid.parquet", part_size="100MB")

    workflow.fit(train_dataset)

    dict_dtypes = {}

    for col in CATEGORICAL_COLUMNS:
        dict_dtypes[col] = np.int64

    for col in LABEL_COLUMNS:
        dict_dtypes[col] = np.float32

    if path.exists(DATA_DIR + "train"):
        shutil.rmtree(os.path.join(DATA_DIR, "train"))
    if path.exists(DATA_DIR + "valid"):
        shutil.rmtree(os.path.join(DATA_DIR, "valid"))

    workflow.transform(train_dataset).to_parquet(
        output_path=DATA_DIR + "train/",
        shuffle=nvt.io.Shuffle.PER_PARTITION,
        cats=CATEGORICAL_COLUMNS,
        labels=LABEL_COLUMNS,
        dtypes=dict_dtypes,
    )
    workflow.transform(valid_dataset).to_parquet(
        output_path=DATA_DIR + "valid/",
        shuffle=False,
        cats=CATEGORICAL_COLUMNS,
        labels=LABEL_COLUMNS,
        dtypes=dict_dtypes,
    )

    # Train with HugeCTR
    embeddings = get_embedding_sizes(workflow)
    total_cardinality = 0
    slot_sizes = []
    for column in CATEGORICAL_COLUMNS:
        slot_sizes.append(embeddings[column][0])
        total_cardinality += embeddings[column][0]

    test_data_path = DATA_DIR + "test/"
    if path.exists(test_data_path):
        shutil.rmtree(test_data_path)

    os.mkdir(test_data_path)

    if path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)

    os.makedirs(TRAIN_DIR)

    sample_data = cudf.read_parquet(DATA_DIR + "valid.parquet", num_rows=TEST_N_ROWS)
    sample_data.to_csv(test_data_path + "data.csv")

    sample_data_trans = nvt.workflow._transform_partition(sample_data, [workflow.output_node])

    dense_features, embedding_columns, row_ptrs = _convert(sample_data_trans, slot_sizes)

    _run_model(slot_sizes, total_cardinality)

    if path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    os.mkdir(TEMP_DIR)

    file_names = glob.iglob(os.path.join(os.getcwd(), "*.model"))
    for files in file_names:
        shutil.move(files, TEMP_DIR)

    hugectr_params = dict()
    hugectr_params["config"] = NETWORK_FILE
    hugectr_params["slots"] = len(slot_sizes)
    hugectr_params["max_nnz"] = len(slot_sizes)
    hugectr_params["embedding_vector_size"] = 16
    hugectr_params["n_outputs"] = 1

    export_hugectr_ensemble(
        workflow=workflow,
        hugectr_model_path=TEMP_DIR,
        hugectr_params=hugectr_params,
        name=MODEL_NAME,
        output_path=MODEL_DIR,
        label_columns=["rating"],
        cats=CATEGORICAL_COLUMNS,
        max_batch_size=64,
    )

    shutil.rmtree(TEMP_DIR)
    _predict(dense_features, embedding_columns, row_ptrs, hugectr_params["config"], MODEL_NAME)


@pytest.mark.parametrize("n_rows", [64, 58, 11, 1])
@pytest.mark.parametrize("err_tol", [0.00001])
def test_inference(n_rows, err_tol):
    warnings.simplefilter("ignore")

    data_path = DATA_DIR + "test/data.csv"
    output_path = DATA_DIR + "test/output.csv"
    ps_file = TRAIN_DIR + "ps.json"

    workflow_path = MODEL_DIR + MODEL_NAME + "_nvt/1/workflow"

    _write_ps_hugectr(ps_file, MODEL_NAME, SPARSE_FILES, DENSE_FILE, NETWORK_FILE)

    with test_utils.run_triton_server(
        os.path.expanduser(MODEL_DIR),
        MODEL_NAME + "_ens",
        TRITON_SERVER_PATH,
        TRITON_DEVICE_ID,
        "hugectr",
        ps_file,
    ) as client:
        diff, run_time = _run_query(
            client,
            n_rows,
            MODEL_NAME + "_ens",
            workflow_path,
            data_path,
            output_path,
            "OUTPUT0",
            CATEGORICAL_COLUMNS,
            "hugectr",
        )
    assert (diff < err_tol).all()
    benchmark_results = []
    result = create_bench_result(
        "test_nvt_hugectr_inference", [("n_rows", n_rows)], run_time, "datetime"
    )
    benchmark_results.append(result)
    # send_results(asv_db, bench_info, benchmark_results)


def _run_model(slot_sizes, total_cardinality):

    solver = hugectr.CreateSolver(
        vvgpu=[[0]],
        batchsize=2048,
        batchsize_eval=2048,
        max_eval_batches=160,
        i64_input_key=True,
        use_mixed_precision=False,
        repeat_dataset=True,
    )

    reader = hugectr.DataReaderParams(
        data_reader_type=hugectr.DataReaderType_t.Parquet,
        source=[DATA_DIR + "train/_file_list.txt"],
        eval_source=DATA_DIR + "valid/_file_list.txt",
        check_type=hugectr.Check_t.Non,
    )

    optimizer = hugectr.CreateOptimizer(optimizer_type=hugectr.Optimizer_t.Adam)
    model = hugectr.Model(solver, reader, optimizer)

    model.add(
        hugectr.Input(
            label_dim=1,
            label_name="label",
            dense_dim=0,
            dense_name="dense",
            data_reader_sparse_param_array=[
                hugectr.DataReaderSparseParam("data1", len(slot_sizes) + 1, True, len(slot_sizes))
            ],
        )
    )

    model.add(
        hugectr.SparseEmbedding(
            embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
            workspace_size_per_gpu_in_mb=107,
            embedding_vec_size=16,
            combiner="sum",
            sparse_embedding_name="sparse_embedding1",
            bottom_name="data1",
            slot_size_array=slot_sizes,
            optimizer=optimizer,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Reshape,
            bottom_names=["sparse_embedding1"],
            top_names=["reshape1"],
            leading_dim=48,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.InnerProduct,
            bottom_names=["reshape1"],
            top_names=["fc1"],
            num_output=128,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.ReLU,
            bottom_names=["fc1"],
            top_names=["relu1"],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.InnerProduct,
            bottom_names=["relu1"],
            top_names=["fc2"],
            num_output=128,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.ReLU,
            bottom_names=["fc2"],
            top_names=["relu2"],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.InnerProduct,
            bottom_names=["relu2"],
            top_names=["fc3"],
            num_output=1,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
            bottom_names=["fc3", "label"],
            top_names=["loss"],
        )
    )
    model.compile()
    model.summary()
    model.fit(max_iter=2000, display=100, eval_interval=200, snapshot=1900)
    model.graph_to_json(graph_config_file=NETWORK_FILE)


def _predict(dense_features, embedding_columns, row_ptrs, config_file, model_name):
    inference_params = InferenceParams(
        model_name=model_name,
        max_batchsize=64,
        hit_rate_threshold=0.5,
        dense_model_file=DENSE_FILE,
        sparse_model_files=[SPARSE_FILES],
        device_id=0,
        use_gpu_embedding_cache=True,
        cache_size_percentage=0.1,
        i64_input_key=True,
        use_mixed_precision=False,
    )
    inference_session = CreateInferenceSession(config_file, inference_params)
    output = inference_session.predict(dense_features, embedding_columns, row_ptrs)  # , True)

    test_data_path = DATA_DIR + "test/"
    embedding_columns_df = pd.DataFrame()
    embedding_columns_df["embedding_columns"] = embedding_columns
    embedding_columns_df.to_csv(test_data_path + "embedding_columns.csv")

    row_ptrs_df = pd.DataFrame()
    row_ptrs_df["row_ptrs"] = row_ptrs
    row_ptrs_df.to_csv(test_data_path + "row_ptrs.csv")

    output_df = pd.DataFrame()
    output_df["output"] = output
    output_df.to_csv(test_data_path + "output.csv")


def _convert(data, slot_size_array):
    categorical_dim = len(CATEGORICAL_COLUMNS)
    batch_size = data.shape[0]

    offset = np.insert(np.cumsum(slot_size_array), 0, 0)[:-1].tolist()
    data[CATEGORICAL_COLUMNS] += offset
    cat = data[CATEGORICAL_COLUMNS].values.reshape(1, batch_size * categorical_dim).tolist()[0]

    row_ptrs = list(range(batch_size * categorical_dim + 1))
    dense = []

    return dense, cat, row_ptrs


def _write_ps_hugectr(output_file, model_name, sparse_files, dense_file, network_file):
    config = json.dumps(
        {
            "supportlonglong": True,
            "models": [
                {
                    "model": model_name,
                    "sparse_files": [sparse_files],
                    "dense_file": dense_file,
                    "network_file": network_file,
                }
            ],
        }
    )

    config = json.loads(config)
    with open(output_file, "w") as f:
        json.dump(config, f)
