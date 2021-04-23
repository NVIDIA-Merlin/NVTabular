import gc
import glob
import json
import os
import shutil
from os import path

import cudf
import hugectr
import numpy as np
from mpi4py import MPI
from sklearn.model_selection import train_test_split

import nvtabular as nvt
from nvtabular.inference.triton import export_hugectr_ensemble
from nvtabular.ops import get_embedding_sizes
from nvtabular.utils import download_file

import cudf
import numpy as np


DIR = "/model/"
BASE_DIR = DIR + "data/"
TEMP_DIR = DIR + "temp_hugectr/"
MODEL_DIR = DIR + "models/"


def test_nvt_hugectr_training():
    download_file(
        "http://files.grouplens.org/datasets/movielens/ml-25m.zip",
        os.path.join(BASE_DIR, "ml-25m.zip"),
    )

    ratings = cudf.read_csv(os.path.join(BASE_DIR, "ml-25m", "ratings.csv"))
    # ratings["new_cat1"] = ratings["userId"] / ratings["movieId"]
    # ratings["new_cat1"] = ratings["new_cat1"].astype("int64")
    ratings.head()

    ratings = ratings.drop("timestamp", axis=1)
    train, valid = train_test_split(ratings, test_size=0.2, random_state=42)

    train.to_parquet(BASE_DIR + "train.parquet")
    valid.to_parquet(BASE_DIR + "valid.parquet")

    del train
    del valid
    gc.collect()

    CATEGORICAL_COLUMNS = ["userId", "movieId"]
    LABEL_COLUMNS = ["rating"]

    cat_features = CATEGORICAL_COLUMNS >> nvt.ops.Categorify(cat_cache="device")
    ratings = nvt.ColumnGroup(["rating"]) >> (lambda col: (col > 3).astype("int8"))
    output = cat_features + ratings

    workflow = nvt.Workflow(output)

    train_dataset = nvt.Dataset(BASE_DIR + "train.parquet", part_size="100MB")
    valid_dataset = nvt.Dataset(BASE_DIR + "valid.parquet", part_size="100MB")

    workflow.fit(train_dataset)

    dict_dtypes = {}

    for col in CATEGORICAL_COLUMNS:
        dict_dtypes[col] = np.int64

    for col in LABEL_COLUMNS:
        dict_dtypes[col] = np.float32

    if path.exists(BASE_DIR + "train"):
        shutil.rmtree(os.path.join(BASE_DIR, "train"))
    if path.exists(BASE_DIR + "valid"):
        shutil.rmtree(os.path.join(BASE_DIR, "valid"))

    workflow.transform(train_dataset).to_parquet(
        output_path=BASE_DIR + "train/",
        shuffle=nvt.io.Shuffle.PER_PARTITION,
        cats=CATEGORICAL_COLUMNS,
        labels=LABEL_COLUMNS,
        dtypes=dict_dtypes,
    )
    workflow.transform(valid_dataset).to_parquet(
        output_path=BASE_DIR + "valid/",
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

    _run_model(slot_sizes, total_cardinality)

    if path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    os.mkdir(TEMP_DIR)

    file_names = glob.iglob(os.path.join(os.getcwd(), "*.model"))
    for files in file_names:
        shutil.move(files, TEMP_DIR)

    _write_model_json(slot_sizes)

    if path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)

    os.mkdir(MODEL_DIR)

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
        name="test_model",
        output_path=MODEL_DIR,
        label_columns=["rating"],
        cats=CATEGORICAL_COLUMNS,
        max_batch_size=64,
    )


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
            source=BASE_DIR + "train/_file_list.txt",
            eval_source=BASE_DIR + "valid/_file_list.txt",
            check_type=hugectr.Check_t.Non,
            label_dim=1,
            label_name="label",
            dense_dim=0,
            dense_name="dense",
            slot_size_array=slot_sizes,
            data_reader_sparse_param_array=[
                hugectr.DataReaderSparseParam(hugectr.DataReaderSparse_t.Distributed, 3, 1, 2)
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
            leading_dim=32,
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


def _write_model_json(slot_sizes):

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
                    "source": "/model/data/train/_file_list.txt",
                    "eval_source": "/model/data/valid/_file_list.txt",
                    "check": "Sum",
                    "label": {"top": "label", "label_dim": 1},
                    "dense": {"top": "dense", "dense_dim": 0},
                    "sparse": [
                        {
                            "top": "data1",
                            "type": "DistributedSlot",
                            "max_feature_num_per_sample": 3,
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
                        "max_vocabulary_size_per_gpu": 219128,
                        "embedding_vec_size": 16,
                        "combiner": 0,
                    },
                },
                {
                    "name": "reshape1",
                    "type": "Reshape",
                    "bottom": "sparse_embedding1",
                    "top": "reshape1",
                    "leading_dim": 32,
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
