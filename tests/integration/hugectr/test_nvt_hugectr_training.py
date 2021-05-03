import gc
import glob
import json
import os
import shutil
from os import path

import cudf
import hugectr
import numpy as np
import pandas as pd
from hugectr.inference import CreateEmbeddingCache, CreateParameterServer, InferenceSession
from mpi4py import MPI  # noqa
from sklearn.model_selection import train_test_split

import nvtabular as nvt
from nvtabular.inference.triton import export_hugectr_ensemble
from nvtabular.ops import get_embedding_sizes
from nvtabular.utils import download_file

DIR = "/model/"
DATA_DIR = DIR + "data/"
TEMP_DIR = DIR + "temp_hugectr/"
MODEL_DIR = DIR + "models/"

CATEGORICAL_COLUMNS = ["userId", "movieId", "new_cat1"]
LABEL_COLUMNS = ["rating"]
TEST_N_ROWS = 64


def test_nvt_hugectr_training():

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

    cat_features = CATEGORICAL_COLUMNS >> nvt.ops.Categorify(cat_cache="device")
    ratings = nvt.ColumnGroup(["rating"]) >> (lambda col: (col > 3).astype("int8"))
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

    sample_data = cudf.read_parquet(DATA_DIR + "valid.parquet", num_rows=TEST_N_ROWS)
    sample_data.to_csv(test_data_path + "data.csv")

    sample_data_trans = nvt.workflow._transform_partition(sample_data, [workflow.column_group])

    dense_features, embedding_columns, row_ptrs = _convert(sample_data_trans, slot_sizes)

    _run_model(slot_sizes, total_cardinality)

    if path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    os.mkdir(TEMP_DIR)

    file_names = glob.iglob(os.path.join(os.getcwd(), "*.model"))
    for files in file_names:
        shutil.move(files, TEMP_DIR)

    _write_model_json(slot_sizes, total_cardinality)

    if path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)

    os.mkdir(MODEL_DIR)

    model_name = "test_model"
    hugectr_params = dict()
    hugectr_params["config"] = MODEL_DIR + "test_model/1/model.json"
    hugectr_params["slots"] = len(slot_sizes)
    hugectr_params["max_nnz"] = len(slot_sizes)
    hugectr_params["embedding_vector_size"] = 16
    hugectr_params["n_outputs"] = 1

    export_hugectr_ensemble(
        workflow=workflow,
        hugectr_model_path=TEMP_DIR,
        hugectr_params=hugectr_params,
        name=model_name,
        output_path=MODEL_DIR,
        label_columns=["rating"],
        cats=CATEGORICAL_COLUMNS,
        max_batch_size=64,
    )

    shutil.rmtree(TEMP_DIR)

    _predict(dense_features, embedding_columns, row_ptrs, hugectr_params["config"], model_name)


def _run_model(slot_sizes, total_cardinality):

    solver = hugectr.solver_parser_helper(
        vvgpu=[[0]],
        max_iter=2000,
        batchsize=2048,
        display=100,
        eval_interval=200,
        batchsize_eval=2048,
        max_eval_batches=160,
        i64_input_key=True,
        use_mixed_precision=False,
        repeat_dataset=True,
        snapshot=1900,
    )

    optimizer = hugectr.optimizer.CreateOptimizer(
        optimizer_type=hugectr.Optimizer_t.Adam, use_mixed_precision=False
    )
    model = hugectr.Model(solver, optimizer)

    model.add(
        hugectr.Input(
            data_reader_type=hugectr.DataReaderType_t.Parquet,
            source=DATA_DIR + "train/_file_list.txt",
            eval_source=DATA_DIR + "valid/_file_list.txt",
            check_type=hugectr.Check_t.Non,
            label_dim=1,
            label_name="label",
            dense_dim=0,
            dense_name="dense",
            slot_size_array=slot_sizes,
            data_reader_sparse_param_array=[
                hugectr.DataReaderSparseParam(
                    hugectr.DataReaderSparse_t.Distributed, len(slot_sizes) + 1, 1, len(slot_sizes)
                )
            ],
            sparse_names=["data1"],
        )
    )

    model.add(
        hugectr.SparseEmbedding(
            embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
            max_vocabulary_size_per_gpu=total_cardinality,
            embedding_vec_size=16,
            combiner=0,
            sparse_embedding_name="sparse_embedding1",
            bottom_name="data1",
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
    model.fit()


def _write_model_json(slot_sizes, total_cardinality):

    config = json.dumps(
        {
            "inference": {
                "max_batchsize": 64,
                "hit_rate_threshold": 0.6,
                "dense_model_file": MODEL_DIR + "test_model/1/_dense_1900.model",
                "sparse_model_file": MODEL_DIR + "test_model/1/0_sparse_1900.model",
                "label": 1,
                "input_key_type": "I64",
            },
            "layers": [
                {
                    "name": "data",
                    "type": "Data",
                    "format": "Parquet",
                    "slot_size_array": slot_sizes,
                    "source": DATA_DIR + "train/_file_list.txt",
                    "eval_source": DATA_DIR + "valid/_file_list.txt",
                    "check": "Sum",
                    "label": {"top": "label", "label_dim": 1},
                    "dense": {"top": "dense", "dense_dim": 0},
                    "sparse": [
                        {
                            "top": "data1",
                            "type": "DistributedSlot",
                            "max_feature_num_per_sample": len(CATEGORICAL_COLUMNS),
                            "slot_num": len(slot_sizes),
                        }
                    ],
                },
                {
                    "name": "sparse_embedding1",
                    "type": "DistributedSlotSparseEmbeddingHash",
                    "bottom": "data1",
                    "top": "sparse_embedding1",
                    "sparse_embedding_hparam": {
                        "max_vocabulary_size_per_gpu": total_cardinality,
                        "embedding_vec_size": 16,
                        "combiner": 0,
                    },
                },
                {
                    "name": "reshape1",
                    "type": "Reshape",
                    "bottom": "sparse_embedding1",
                    "top": "reshape1",
                    "leading_dim": 48,
                },
                {
                    "name": "fc1",
                    "type": "InnerProduct",
                    "bottom": "reshape1",
                    "top": "fc1",
                    "fc_param": {"num_output": 128},
                },
                {"name": "relu1", "type": "ReLU", "bottom": "fc1", "top": "relu1"},
                {
                    "name": "fc2",
                    "type": "InnerProduct",
                    "bottom": "relu1",
                    "top": "fc2",
                    "fc_param": {"num_output": 128},
                },
                {"name": "relu2", "type": "ReLU", "bottom": "fc2", "top": "relu2"},
                {
                    "name": "fc3",
                    "type": "InnerProduct",
                    "bottom": "relu2",
                    "top": "fc3",
                    "fc_param": {"num_output": 1},
                },
                {"name": "sigmoid", "type": "Sigmoid", "bottom": "fc3", "top": "sigmoid"},
            ],
        }
    )

    config = json.loads(config)

    f = open(os.path.join(TEMP_DIR, "model.json"), "w")
    json.dump(config, f)


def _predict(dense_features, embedding_columns, row_ptrs, config_file, model_name):
    parameter_server = CreateParameterServer([config_file], [model_name], True)
    embedding_cache = CreateEmbeddingCache(
        parameter_server, 0, True, 0.1, config_file, model_name, True
    )
    inference_session = InferenceSession(config_file, 0, embedding_cache)

    output = inference_session.predict(dense_features, embedding_columns, row_ptrs, True)

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

    row_ptrs = [i for i in range(batch_size * categorical_dim + 1)]
    dense = []

    return dense, cat, row_ptrs
