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

import concurrent.futures
import datetime as dt
import glob
import os
import warnings
from distutils.spawn import find_executable

import cudf
import cupy as cp
import numpy as np
import pytest
import tritonclient.grpc as grpcclient

# from benchmark_parsers import send_results
from tritonclient.utils import np_to_triton_dtype

import nvtabular as nvt
import tests.conftest as test_utils

TEST_N_ROWS = 1024
MODEL_DIR = "~/nvt-examples/models/"
DATA_DIR = "~/nvt-examples/data/"
TRITON_SERVER_PATH = find_executable("tritonserver")


# Update TEST_N_ROWS param in test_nvt_tf_trainin.py to test larger sizes
@pytest.mark.parametrize("n_rows", [64, 45, 32, 14, 7, 1])
@pytest.mark.parametrize("err_tol", [0.00001])
def test_nvt_tf_movielens_inference(n_rows, err_tol):

    INPUT_DATA_DIR = os.path.expanduser("~/nvt-examples/movielens/data/")

    warnings.simplefilter("ignore")

    model_name = "test_movielens_tf"
    col_names = ["userId", "movieId"]
    # read in a batch of data to get transforms for
    batch = cudf.read_csv(os.path.join(INPUT_DATA_DIR, "test_data.csv"), nrows=n_rows)[col_names]

    # convert the batch to a triton inputs
    columns = [(col, batch[col]) for col in col_names]
    inputs = []

    col_dtypes = [np.int64, np.int64]
    for i, (name, col) in enumerate(columns):
        d = col.values_host.astype(col_dtypes[i])
        d = d.reshape(len(d), 1)
        inputs.append(grpcclient.InferInput(name, d.shape, np_to_triton_dtype(col_dtypes[i])))
        inputs[i].set_data_from_numpy(d)

    # placeholder variables for the output
    outputs = []
    outputs.append(grpcclient.InferRequestedOutput("output"))
    # make the request
    with grpcclient.InferenceServerClient("localhost:8001") as client:
        response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    output_actual = cudf.read_csv(os.path.join(INPUT_DATA_DIR, "output.csv"), nrows=n_rows)
    output_actual = cp.asnumpy(output_actual["0"].values)
    output_predict = response.as_numpy("output")

    diff = abs(output_actual - output_predict[:, 0])
    assert (diff < err_tol).all()


def _run_rossmann_query(client, n_rows, err_tol):
    workflow_path = os.path.join(os.path.expanduser(MODEL_DIR), "rossmann_nvt/1/workflow")
    data_path = os.path.join(os.path.expanduser(DATA_DIR), "test_inference_rossmann_data.csv")
    actual_output_filename = os.path.join(os.path.expanduser(DATA_DIR), "rossmann_predictions.csv")

    workflow = nvt.Workflow.load(workflow_path)
    batch = cudf.read_csv(data_path, nrows=n_rows)[workflow.column_group.input_column_names]

    columns = [(col, batch[col]) for col in batch.columns]

    inputs = []
    for i, (name, col) in enumerate(columns):
        d = col.values_host.astype(col.dtype)
        d = d.reshape(len(d), 1)
        inputs.append(grpcclient.InferInput(name, d.shape, np_to_triton_dtype(col.dtype)))
        inputs[i].set_data_from_numpy(d)

    outputs = [grpcclient.InferRequestedOutput("tf.math.multiply_1")]
    time_start = dt.datetime.now()
    response = client.infer("rossmann", inputs, request_id="1", outputs=outputs)
    run_time = dt.datetime.now() - time_start

    output_actual = cudf.read_csv(os.path.expanduser(actual_output_filename), nrows=n_rows)
    output_actual = cp.asnumpy(output_actual["0"].values)
    output_predict = response.as_numpy("tf.math.multiply_1")

    diff = abs(output_actual - output_predict[:, 0])
    return diff, run_time


@pytest.mark.parametrize("n_rows", [1024, 1000, 64, 35, 16, 5])
@pytest.mark.parametrize("err_tol", [0.00001])
def test_nvt_tf_rossmann_inference_triton(asv_db, bench_info, n_rows, err_tol):
    with test_utils.run_triton_server(os.path.expanduser(MODEL_DIR), TRITON_SERVER_PATH) as client:
        diff, run_time = _run_rossmann_query(client, n_rows, err_tol)
    assert (diff < err_tol).all()
    # send_results(asv_db, bench_info, "bench_results")


@pytest.mark.parametrize("n_rows", [[1024, 1000, 35, 16]])
@pytest.mark.parametrize("err_tol", [0.00001])
def test_nvt_tf_rossmann_inference_triton_mt(asv_db, bench_info, n_rows, err_tol):
    futures = []
    with test_utils.run_triton_server(os.path.expanduser(MODEL_DIR), TRITON_SERVER_PATH) as client:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for n_row in n_rows:
                futures.append(executor.submit(_run_rossmann_query, client, n_row, err_tol))

    for future in concurrent.futures.as_completed(futures):
        diff, run_time = future.result()
        assert (diff < err_tol).all()
        # send_results(asv_db, bench_info, "bench_results")


@pytest.mark.skipif(TEST_N_ROWS is None, reason="Requires TEST_N_ROWS")
@pytest.mark.skipif(MODEL_DIR is None, reason="Requires MODEL_DIR")
@pytest.mark.skipif(DATA_DIR is None, reason="Requires DATA_DIR")
def test_nvt_tf_rossmann_inference():
    import tensorflow as tf
    from tensorflow import keras

    from nvtabular.loader.tensorflow import KerasSequenceLoader

    workflow_path = os.path.join(os.path.expanduser(MODEL_DIR), "rossmann_nvt/1/workflow")
    model_path = os.path.join(os.path.expanduser(MODEL_DIR), "rossmann_tf/1/model.savedmodel")
    data_path = os.path.join(os.path.expanduser(DATA_DIR), "valid.csv")
    output_dir = os.path.expanduser(DATA_DIR)
    workflow_output_test_file_name = "test_inference_rossmann_data.csv"
    workflow_output_test_trans_file_name = "test_inference_rossmann_data_trans.parquet"
    prediction_file_name = "rossmann_predictions.csv"

    workflow = nvt.Workflow.load(workflow_path)

    sample_data = cudf.read_csv(data_path, nrows=TEST_N_ROWS)
    sample_data.to_csv(os.path.join(output_dir, workflow_output_test_file_name))
    sample_data_trans = nvt.workflow._transform_partition(sample_data, [workflow.column_group])
    sample_data_trans.to_parquet(os.path.join(output_dir, workflow_output_test_trans_file_name))

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

    test_data_trans_path = glob.glob(os.path.join(output_dir, workflow_output_test_trans_file_name))
    EMBEDDING_TABLE_SHAPES = nvt.ops.get_embedding_sizes(workflow)

    categorical_columns = [
        _make_categorical_embedding_column(name, *EMBEDDING_TABLE_SHAPES[name])
        for name in CATEGORICAL_COLUMNS
    ]
    continuous_columns = [
        tf.feature_column.numeric_column(name, (1,)) for name in CONTINUOUS_COLUMNS
    ]

    train_dataset = KerasSequenceLoader(
        test_data_trans_path,  # you could also use a glob pattern
        feature_columns=categorical_columns + continuous_columns,
        batch_size=TEST_N_ROWS,
        label_names=[],
        shuffle=False,
        buffer_size=0.06,  # amount of data, as a fraction of GPU memory, to load at once
    )

    tf_model = keras.models.load_model(model_path, custom_objects={"rmspe_tf": rmspe_tf})

    pred = tf_model.predict(train_dataset)
    cudf_pred = cudf.DataFrame(pred)
    cudf_pred.to_csv(os.path.join(output_dir, prediction_file_name))

    os.remove(os.path.join(output_dir, workflow_output_test_trans_file_name))


def rmspe_tf(y_true, y_pred):
    import tensorflow as tf

    y_true = tf.exp(y_true) - 1
    y_pred = tf.exp(y_pred) - 1

    percent_error = (y_true - y_pred) / y_true
    return tf.sqrt(tf.reduce_mean(percent_error ** 2))


def _make_categorical_embedding_column(name, dictionary_size, embedding_dim):
    import tensorflow as tf

    return tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity(name, dictionary_size), embedding_dim
    )
