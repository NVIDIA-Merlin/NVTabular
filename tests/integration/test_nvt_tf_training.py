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

import glob
import math

# External dependencies
import os
import shutil
from os import path
from os.path import dirname, realpath

import cudf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from test_notebooks import _run_notebook

import nvtabular as nvt
from nvtabular import ops
from nvtabular.inference.triton import export_tensorflow_ensemble
from nvtabular.utils import download_file

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

    sample_data = cudf.read_parquet(
        os.path.join(INPUT_DATA_DIR, "valid.parquet"), num_rows=TEST_N_ROWS
    )
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
    from nvtabular.framework_utils.tensorflow import layers
    from nvtabular.loader.tensorflow import KerasSequenceLoader, KerasSequenceValidater

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

    model.fit(train_dataset_tf, callbacks=[validation_callback], epochs=1)

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

    export_tensorflow_ensemble(model, workflow, MODEL_NAME_ENSEMBLE, MODEL_PATH, ["rating"])


def test_nvt_tf_rossmann_training(tmpdir):

    TEST_PATH = dirname(dirname(realpath(__file__)))
    DATA_START = os.path.expanduser("~/nvt-examples/rossmann/")
    DATA_DIR = os.path.join(DATA_START, "data")
    input_path = os.path.join(DATA_START, "input")
    output_path = os.path.join(DATA_START, "output")

    notebookpre_path = os.path.join(
        dirname(TEST_PATH), "examples/tabular-data-rossmann", "01-Download-Convert.ipynb"
    )

    _run_notebook(tmpdir, notebookpre_path, DATA_DIR, input_path, gpu_id="0", clean_up=False)

    DATA_DIR = os.path.join(DATA_START, "input/")

    CATEGORICAL_COLUMNS = [
        "Store",
        "DayOfWeek",
        "Year",
        "Month",
        "Day",
        "StateHoliday",
        "CompetitionMonthsOpen",
        "Promo2Weeks",
        "StoreType",
        "Assortment",
        "PromoInterval",
        "CompetitionOpenSinceYear",
        "Promo2SinceYear",
        "State",
        "Week",
        "Events",
        "Promo_fw",
        "Promo_bw",
        "StateHoliday_fw",
        "StateHoliday_bw",
        "SchoolHoliday_fw",
        "SchoolHoliday_bw",
    ]

    CONTINUOUS_COLUMNS = [
        "CompetitionDistance",
        "Max_TemperatureC",
        "Mean_TemperatureC",
        "Min_TemperatureC",
        "Max_Humidity",
        "Mean_Humidity",
        "Min_Humidity",
        "Max_Wind_SpeedKm_h",
        "Mean_Wind_SpeedKm_h",
        "CloudCover",
        "trend",
        "trend_DE",
        "AfterStateHoliday",
        "BeforeStateHoliday",
        "Promo",
        "SchoolHoliday",
    ]

    LABEL_COLUMNS = ["Sales"]

    COLUMNS = CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS + LABEL_COLUMNS

    TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
    VALID_PATH = os.path.join(DATA_DIR, "valid.csv")
    
    cat_features = CATEGORICAL_COLUMNS >> ops.Categorify()
    cont_features = CONTINUOUS_COLUMNS >> ops.FillMissing() >> ops.Normalize()
    label_feature = LABEL_COLUMNS >> ops.LogOp()

    proc = nvt.Workflow(cat_features + cont_features + label_feature)

    train_dataset = nvt.Dataset(TRAIN_PATH)
    valid_dataset = nvt.Dataset(VALID_PATH)

    proc.fit(train_dataset)

    PREPROCESS_DIR = os.path.join(DATA_DIR, "ross_pre/")
    PREPROCESS_DIR_TRAIN = os.path.join(PREPROCESS_DIR, "train")
    PREPROCESS_DIR_VALID = os.path.join(PREPROCESS_DIR, "valid")

    if path.exists(PREPROCESS_DIR):
        shutil.rmtree(PREPROCESS_DIR)

    if path.exists(PREPROCESS_DIR_TRAIN):
        shutil.rmtree(PREPROCESS_DIR_TRAIN)

    if path.exists(PREPROCESS_DIR_VALID):
        shutil.rmtree(PREPROCESS_DIR_VALID)

    proc.transform(train_dataset).to_parquet(PREPROCESS_DIR_TRAIN, shuffle=nvt.io.Shuffle.PER_WORKER)
    proc.transform(valid_dataset).to_parquet(PREPROCESS_DIR_VALID, shuffle=None)

    sample_data = cudf.read_csv(VALID_PATH, nrows=TEST_N_ROWS)
    sample_data.to_csv(os.path.join(DATA_DIR, "test_data.csv"))
    sample_data_trans = nvt.workflow._transform_partition(sample_data, [proc.column_group])
    sample_data_trans.to_parquet(os.path.join(DATA_DIR, "test_data_trans.parquet"))

    EMBEDDING_TABLE_SHAPES = nvt.ops.get_embedding_sizes(proc)

    EMBEDDING_DROPOUT_RATE = 0.04
    DROPOUT_RATES = [0.001, 0.01]
    HIDDEN_DIMS = [1000, 500]
    BATCH_SIZE = 65536
    LEARNING_RATE = 0.001
    EPOCHS = 25

    # TODO: Calculate on the fly rather than recalling from previous analysis.
    MAX_SALES_IN_TRAINING_SET = 38722.0
    MAX_LOG_SALES_PREDICTION = 1.2 * math.log(MAX_SALES_IN_TRAINING_SET + 1.0)

    TRAIN_PATHS = sorted(glob.glob(os.path.join(PREPROCESS_DIR_TRAIN, "*.parquet")))
    VALID_PATHS = sorted(glob.glob(os.path.join(PREPROCESS_DIR_VALID, "*.parquet")))

    os.environ["TF_MEMORY_ALLOCATION"] = "8192"  # explicit MB
    os.environ["TF_MEMORY_ALLOCATION"] = "0.5"  # fraction of free memory
    from nvtabular.loader.tensorflow import KerasSequenceLoader, KerasSequenceValidater

    # instantiate our columns
    categorical_columns = [
        _make_categorical_embedding_column(name, *EMBEDDING_TABLE_SHAPES[name])
        for name in CATEGORICAL_COLUMNS
    ]
    continuous_columns = [tf.feature_column.numeric_column(name, (1,)) for name in CONTINUOUS_COLUMNS]

    # feed them to our datasets
    train_dataset = KerasSequenceLoader(
        TRAIN_PATHS,  # you could also use a glob pattern
        feature_columns=categorical_columns + continuous_columns,
        batch_size=BATCH_SIZE,
        label_names=LABEL_COLUMNS,
        shuffle=True,
        buffer_size=0.06,  # amount of data, as a fraction of GPU memory, to load at once
    )

    valid_dataset = KerasSequenceLoader(
        VALID_PATHS,  # you could also use a glob pattern
        feature_columns=categorical_columns + continuous_columns,
        batch_size=BATCH_SIZE * 4,
        label_names=LABEL_COLUMNS,
        shuffle=False,
        buffer_size=0.06,  # amount of data, as a fraction of GPU memory, to load at once
    )

    # DenseFeatures layer needs a dictionary of {feature_name: input}
    categorical_inputs = {}
    for column_name in CATEGORICAL_COLUMNS:
        categorical_inputs[column_name] = tf.keras.Input(name=column_name, shape=(1,), dtype=tf.int64)
    categorical_embedding_layer = tf.keras.layers.DenseFeatures(categorical_columns)
    categorical_x = categorical_embedding_layer(categorical_inputs)
    categorical_x = tf.keras.layers.Dropout(EMBEDDING_DROPOUT_RATE)(categorical_x)

    # Just concatenating continuous, so can use a list
    continuous_inputs = []
    for column_name in CONTINUOUS_COLUMNS:
        continuous_inputs.append(tf.keras.Input(name=column_name, shape=(1,), dtype=tf.float32))
    continuous_embedding_layer = tf.keras.layers.Concatenate(axis=1)
    continuous_x = continuous_embedding_layer(continuous_inputs)
    continuous_x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.1)(continuous_x)

    # concatenate and build MLP
    x = tf.keras.layers.Concatenate(axis=1)([categorical_x, continuous_x])
    for dim, dropout_rate in zip(HIDDEN_DIMS, DROPOUT_RATES):
        x = tf.keras.layers.Dense(dim, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(1, activation="linear")(x)

    # TODO: Initialize model weights to fix saturation issues.
    # For now, we'll just scale the output of our model directly before
    # hitting the sigmoid.
    x = 0.1 * x

    x = MAX_LOG_SALES_PREDICTION * tf.keras.activations.sigmoid(x)

    # combine all our inputs into a single list
    # (note that you can still use .fit, .predict, etc. on a dict
    # that maps input tensor names to input values)
    inputs = list(categorical_inputs.values()) + continuous_inputs
    tf_model = tf.keras.Model(inputs=inputs, outputs=x)
 
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    tf_model.compile(optimizer, "mse", metrics=[_rmspe_tf])

    validation_callback = KerasSequenceValidater(valid_dataset)

    tf_model.fit(
        train_dataset,
        callbacks=[validation_callback],
        epochs=EPOCHS,
    )

    sample_data_trans = KerasSequenceLoader(
        os.path.join(DATA_DIR, "test_data_trans.parquet"),
        batch_size=TEST_N_ROWS,
        label_names=LABEL_COLUMNS,
        feature_columns=categorical_columns + continuous_columns,
        engine="parquet",
        shuffle=False,
        buffer_size=0.06,
        parts_per_chunk=1,
    )

    pred = tf_model.predict(sample_data_trans)
    cudf_pred = cudf.DataFrame(pred)
    cudf_pred.to_csv(os.path.join(DATA_DIR, "output.csv"))

    export_tensorflow_ensemble(tf_model, proc, 'test_rossmann_tf', os.path.expanduser("~/nvt-examples/models/"), LABEL_COLUMNS)
    
def _rmspe_tf(y_true, y_pred):
    # map back into "true" space by undoing transform
    y_true = tf.exp(y_true) - 1
    y_pred = tf.exp(y_pred) - 1

    percent_error = (y_true - y_pred) / y_true
    return tf.sqrt(tf.reduce_mean(percent_error ** 2))

def _make_categorical_embedding_column(name, dictionary_size, embedding_dim):
    return tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity(name, dictionary_size), embedding_dim
    )
