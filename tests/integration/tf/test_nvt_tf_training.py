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

# External dependencies
import os
import cudf  # cuDF is an implementation of Pandas-like Dataframe on GPU
from sklearn.model_selection import train_test_split
from nvtabular.utils import download_file
import shutil
import numpy as np
import nvtabular as nvt
from os import path
import glob
import tensorflow as tf

TEST_N_ROWS = 64

def test_nvt_tf_movielens_training():

    INPUT_DATA_DIR = os.path.expanduser("~/nvt-examples/movielens/data/")

    download_file(
        "http://files.grouplens.org/datasets/movielens/ml-25m.zip",
        os.path.join(INPUT_DATA_DIR, "ml-25m.zip"),
    )

    movies = cudf.read_csv(os.path.join(INPUT_DATA_DIR, "ml-25m/movies.csv"))
    movies["genres"] = movies["genres"].str.split("|")
    movies = movies.drop("title", axis=1)

    movies.to_parquet(os.path.join(INPUT_DATA_DIR, "movies_converted.parquet"))
    ratings = cudf.read_csv(os.path.join(INPUT_DATA_DIR, "ml-25m", "ratings.csv"))

    ratings = ratings.drop("timestamp", axis=1)
    train, valid = train_test_split(ratings, test_size=0.2, random_state=42)

    train.to_parquet(os.path.join(INPUT_DATA_DIR, "train.parquet"))
    valid.to_parquet(os.path.join(INPUT_DATA_DIR, "valid.parquet"))

    movies = cudf.read_parquet(os.path.join(INPUT_DATA_DIR, "movies_converted.parquet"))
    
    CATEGORICAL_COLUMNS = ["userId", "movieId"]
    LABEL_COLUMNS = ["rating"]

    joined = ["userId", "movieId"] >> nvt.ops.JoinExternal(movies, on=["movieId"])

    cat_features = joined >> nvt.ops.Categorify()

    ratings = nvt.ColumnGroup(["rating"]) >> (lambda col: (col > 3).astype("int8"))
    output = cat_features + ratings

    workflow = nvt.Workflow(output)

    dict_dtypes = {}

    for col in CATEGORICAL_COLUMNS:
        dict_dtypes[col] = np.int64

    for col in LABEL_COLUMNS:
        dict_dtypes[col] = np.float32

    train_dataset = nvt.Dataset([os.path.join(INPUT_DATA_DIR, "train.parquet")], part_size="100MB")
    valid_dataset = nvt.Dataset([os.path.join(INPUT_DATA_DIR, "valid.parquet")], part_size="100MB")

    workflow.fit(train_dataset)

    if path.exists(os.path.join(INPUT_DATA_DIR, "train")):
        shutil.rmtree(os.path.join(INPUT_DATA_DIR, "train"))
    if path.exists(os.path.join(INPUT_DATA_DIR, "valid")):
        shutil.rmtree(os.path.join(INPUT_DATA_DIR, "valid"))

    workflow.transform(train_dataset).to_parquet(
        output_path=os.path.join(INPUT_DATA_DIR, "train"),
        shuffle=nvt.io.Shuffle.PER_PARTITION,
        cats=["userId", "movieId"],
        labels=["rating"],
        dtypes=dict_dtypes,
    )

    workflow.transform(valid_dataset).to_parquet(
        output_path=os.path.join(INPUT_DATA_DIR, "valid"),
        shuffle=False,
        cats=["userId", "movieId"],
        labels=["rating"],
        dtypes=dict_dtypes,
    )

    sample_data = cudf.read_parquet(os.path.join(INPUT_DATA_DIR, "valid.parquet"), num_rows=TEST_N_ROWS)
    sample_data.to_csv(os.path.join(INPUT_DATA_DIR, "test_data.csv"))
    sample_data_trans = nvt.workflow._transform_partition(sample_data, [workflow.column_group])
    sample_data_trans.to_parquet(os.path.join(INPUT_DATA_DIR, "test_data_trans.parquet"))
    
    workflow.save(os.path.join(INPUT_DATA_DIR, "workflow"))

    MODEL_BASE_DIR = os.environ.get("MODEL_BASE_DIR", os.path.expanduser("~/nvt-examples/"))
    BATCH_SIZE = 1024 * 32  # Batch Size
    CATEGORICAL_COLUMNS = ["movieId", "userId"]  # Single-hot
    CATEGORICAL_MH_COLUMNS = ["genres"]  # Multi-hot
    NUMERIC_COLUMNS = []

    # Output from ETL-with-NVTabular
    TRAIN_PATHS = sorted(glob.glob(os.path.join(INPUT_DATA_DIR, "train", "*.parquet")))
    VALID_PATHS = sorted(glob.glob(os.path.join(INPUT_DATA_DIR, "valid", "*.parquet")))

    workflow = nvt.Workflow.load(os.path.join(INPUT_DATA_DIR, "workflow"))

    EMBEDDING_TABLE_SHAPES, MH_EMBEDDING_TABLE_SHAPES = nvt.ops.get_embedding_sizes(workflow)
    EMBEDDING_TABLE_SHAPES.update(MH_EMBEDDING_TABLE_SHAPES)

    os.environ["TF_MEMORY_ALLOCATION"] = "0.7"  # fraction of free memory
    from nvtabular.loader.tensorflow import KerasSequenceLoader, KerasSequenceValidater
    from nvtabular.framework_utils.tensorflow import layers

    train_dataset_tf = KerasSequenceLoader(
        TRAIN_PATHS,  # you could also use a glob pattern
        batch_size=BATCH_SIZE,
        label_names=["rating"],
        cat_names=CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS,
        cont_names=NUMERIC_COLUMNS,
        engine="parquet",
        shuffle=True,
        buffer_size=0.06,  # how many batches to load at once
        parts_per_chunk=1,
    )

    valid_dataset_tf = KerasSequenceLoader(
        VALID_PATHS,  # you could also use a glob pattern
        batch_size=BATCH_SIZE,
        label_names=["rating"],
        cat_names=CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS,
        cont_names=NUMERIC_COLUMNS,
        engine="parquet",
        shuffle=False,
        buffer_size=0.06,
        parts_per_chunk=1,
    )  
  
    inputs = {}  # tf.keras.Input placeholders for each feature to be used
    emb_layers = []  # output of all embedding layers, which will be concatenated

    for col in CATEGORICAL_COLUMNS:
        inputs[col] = tf.keras.Input(name=col, dtype=tf.int32, shape=(1,))
    # Note that we need two input tensors for multi-hot categorical features
    for col in CATEGORICAL_MH_COLUMNS:
        inputs[col + "__values"] = tf.keras.Input(name=f"{col}__values", dtype=tf.int64, shape=(1,))
        inputs[col + "__nnzs"] = tf.keras.Input(name=f"{col}__nnzs", dtype=tf.int64, shape=(1,))
    
    for col in CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS:
        emb_layers.append(
            tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_identity(
                    col, EMBEDDING_TABLE_SHAPES[col][0]
                ),  # Input dimension (vocab size)
                EMBEDDING_TABLE_SHAPES[col][1],  # Embedding output dimension
            )
        )

    emb_layer = layers.DenseFeatures(emb_layers)
    x_emb_output = emb_layer(inputs)

    x = tf.keras.layers.Dense(128, activation="relu")(x_emb_output)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile("sgd", "binary_crossentropy")

    validation_callback = KerasSequenceValidater(valid_dataset_tf)

    history = model.fit(train_dataset_tf, callbacks=[validation_callback], epochs=1)

    sample_data_trans = KerasSequenceLoader(
        os.path.join(INPUT_DATA_DIR, "test_data_trans.parquet"),
        batch_size=TEST_N_ROWS,
        label_names=["rating"],
        cat_names=CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS,
        cont_names=NUMERIC_COLUMNS,
        engine="parquet",
        shuffle=False,
        buffer_size=0.06,
        parts_per_chunk=1,
    )  
   
    pred = model.predict(sample_data_trans)
    cudf_pred = cudf.DataFrame(pred)
    cudf_pred.to_csv(os.path.join(INPUT_DATA_DIR, "output.csv"))

    MODEL_NAME_TF = os.environ.get("MODEL_NAME_TF", "movielens_tf")
    MODEL_PATH_TEMP_TF = os.path.join(MODEL_BASE_DIR, MODEL_NAME_TF, "1/model.savedmodel")

    model.save(MODEL_PATH_TEMP_TF)

    workflow = nvt.Workflow.load(os.path.join(INPUT_DATA_DIR, "workflow"))

    workflow.output_dtypes["userId"] = "int32"
    workflow.output_dtypes["movieId"] = "int32"

    MODEL_NAME_ENSEMBLE = "test_movielens_tf"
    # model path to save the models
    MODEL_PATH = os.path.join(MODEL_BASE_DIR, "models")

    from nvtabular.inference.triton import export_tensorflow_ensemble

    export_tensorflow_ensemble(model, workflow, MODEL_NAME_ENSEMBLE, MODEL_PATH, ["rating"])


