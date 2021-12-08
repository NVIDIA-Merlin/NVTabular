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
import glob
import os
from distutils.spawn import find_executable

import cudf
import pytest
from common.parsers.benchmark_parsers import create_bench_result
from common.utils import _run_query

import nvtabular as nvt
import tests.conftest as test_utils

TEST_N_ROWS = 1024
MODEL_DIR = "/model/models/"
DATA_DIR = "/raid/data/"
DATA_DIR_MOVIELENS = "/raid/data/movielens/data/"
TRITON_SERVER_PATH = find_executable("tritonserver")
TRITON_DEVICE_ID = "1"


# Update TEST_N_ROWS param in test_nvt_tf_trainin.py to test larger sizes
@pytest.mark.parametrize("n_rows", [1024, 1000, 64, 35, 16, 5])
@pytest.mark.parametrize("err_tol", [0.00001])
def test_nvt_tf_movielens_inference_triton(asv_db, bench_info, n_rows, err_tol):
    with test_utils.run_triton_server(
        os.path.expanduser(MODEL_DIR),
        "movielens",
        TRITON_SERVER_PATH,
        TRITON_DEVICE_ID,
        "tensorflow",
    ) as client:
        diff, run_time = _run_movielens_query(client, n_rows)

        assert (diff < err_tol).all()
        benchmark_results = []
        result = create_bench_result(
            "test_nvt_tf_movielens_inference_triton", [("n_rows", n_rows)], run_time, "datetime"
        )
        benchmark_results.append(result)
        # send_results(asv_db, bench_info, benchmark_results)


@pytest.mark.parametrize("n_rows", [[1024, 1000, 35, 16]])
@pytest.mark.parametrize("err_tol", [0.00001])
def test_nvt_tf_movielens_inference_triton_mt(asv_db, bench_info, n_rows, err_tol):
    futures = []
    with test_utils.run_triton_server(
        os.path.expanduser(MODEL_DIR),
        "movielens",
        TRITON_SERVER_PATH,
        TRITON_DEVICE_ID,
        "tensorflow",
    ) as client:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for n_row in n_rows:
                futures.append(executor.submit(_run_movielens_query, client, n_row))

    for future in concurrent.futures.as_completed(futures):
        diff, run_time = future.result()
        assert (diff < err_tol).all()
        benchmark_results = []
        result = create_bench_result(
            "test_nvt_tf_movielens_inference_triton_mt", [("n_rows", n_rows)], run_time, "datetime"
        )
        benchmark_results.append(result)
        # send_results(asv_db, bench_info, benchmark_results)


@pytest.mark.skipif(TEST_N_ROWS is None, reason="Requires TEST_N_ROWS")
@pytest.mark.skipif(MODEL_DIR is None, reason="Requires MODEL_DIR")
@pytest.mark.skipif(DATA_DIR is None, reason="Requires DATA_DIR")
def test_nvt_tf_movielens_inference():
    from tensorflow import keras

    from nvtabular.loader.tensorflow import KerasSequenceLoader

    workflow_path = os.path.join(os.path.expanduser(MODEL_DIR), "movielens_nvt/1/workflow")
    model_path = os.path.join(os.path.expanduser(MODEL_DIR), "movielens_tf/1/model.savedmodel")
    data_path = os.path.join(os.path.expanduser(DATA_DIR), "movielens/data/valid.parquet")
    output_dir = os.path.join(os.path.expanduser(DATA_DIR), "movielens/")
    workflow_output_test_file_name = "test_inference_movielens_data.csv"
    workflow_output_test_trans_file_name = "test_inference_movielens_data_trans.parquet"
    prediction_file_name = "movielens_predictions.csv"

    workflow = nvt.Workflow.load(workflow_path)

    sample_data = cudf.read_parquet(data_path, nrows=TEST_N_ROWS)
    sample_data.to_csv(os.path.join(output_dir, workflow_output_test_file_name))
    sample_data_trans = nvt.workflow.workflow._transform_partition(
        sample_data, [workflow.output_node]
    )
    sample_data_trans.to_parquet(os.path.join(output_dir, workflow_output_test_trans_file_name))

    CATEGORICAL_COLUMNS = ["movieId", "userId"]  # Single-hot
    CATEGORICAL_MH_COLUMNS = ["genres"]  # Multi-hot
    NUMERIC_COLUMNS = []

    test_data_trans_path = glob.glob(os.path.join(output_dir, workflow_output_test_trans_file_name))

    train_dataset = KerasSequenceLoader(
        test_data_trans_path,  # you could also use a glob pattern
        batch_size=TEST_N_ROWS,
        label_names=[],
        cat_names=CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS,
        cont_names=NUMERIC_COLUMNS,
        engine="parquet",
        shuffle=False,
        buffer_size=0.06,  # how many batches to load at once
        parts_per_chunk=1,
    )

    tf_model = keras.models.load_model(model_path)

    pred = tf_model.predict(train_dataset)
    cudf_pred = cudf.DataFrame(pred)
    cudf_pred.to_csv(os.path.join(output_dir, prediction_file_name))

    os.remove(os.path.join(output_dir, workflow_output_test_trans_file_name))


@pytest.mark.parametrize("n_rows", [1024, 1000, 64, 35, 16, 5])
@pytest.mark.parametrize("err_tol", [0.00001])
def test_nvt_tf_rossmann_inference_triton(asv_db, bench_info, n_rows, err_tol):
    with test_utils.run_triton_server(
        os.path.expanduser(MODEL_DIR),
        "rossmann",
        TRITON_SERVER_PATH,
        TRITON_DEVICE_ID,
        "tensorflow",
    ) as client:
        diff, run_time = _run_rossmann_query(client, n_rows)

        assert (diff < err_tol).all()
        benchmark_results = []
        result = create_bench_result(
            "test_nvt_tf_rossmann_inference_triton", [("n_rows", n_rows)], run_time, "datetime"
        )
        benchmark_results.append(result)
        # send_results(asv_db, bench_info, benchmark_results)


@pytest.mark.parametrize("n_rows", [[1024, 1000, 35, 16, 5]])
@pytest.mark.parametrize("err_tol", [0.00001])
def test_nvt_tf_rossmann_inference_triton_mt(asv_db, bench_info, n_rows, err_tol):
    futures = []
    with test_utils.run_triton_server(
        os.path.expanduser(MODEL_DIR),
        "rossmann",
        TRITON_SERVER_PATH,
        TRITON_DEVICE_ID,
        "tensorflow",
    ) as client:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for n_row in n_rows:
                futures.append(executor.submit(_run_rossmann_query, client, n_row))

    for future in concurrent.futures.as_completed(futures):
        diff, run_time = future.result()
        assert (diff < err_tol).all()
        benchmark_results = []
        result = create_bench_result(
            "test_nvt_tf_rossmann_inference_triton_mt", [("n_rows", n_rows)], run_time, "datetime"
        )
        benchmark_results.append(result)
        # send_results(asv_db, bench_info, benchmark_results)


@pytest.mark.skipif(TEST_N_ROWS is None, reason="Requires TEST_N_ROWS")
@pytest.mark.skipif(MODEL_DIR is None, reason="Requires MODEL_DIR")
@pytest.mark.skipif(DATA_DIR is None, reason="Requires DATA_DIR")
def test_nvt_tf_rossmann_inference():
    import tensorflow as tf
    from tensorflow import keras

    from nvtabular.loader.tensorflow import KerasSequenceLoader

    workflow_path = os.path.join(os.path.expanduser(MODEL_DIR), "rossmann_nvt/1/workflow")
    model_path = os.path.join(os.path.expanduser(MODEL_DIR), "rossmann_tf/1/model.savedmodel")
    data_path = os.path.join(os.path.expanduser(DATA_DIR), "rossman/input/valid.csv")
    output_dir = os.path.join(os.path.expanduser(DATA_DIR), "rossman/")
    workflow_output_test_file_name = "test_inference_rossmann_data.csv"
    workflow_output_test_trans_file_name = "test_inference_rossmann_data_trans.parquet"
    prediction_file_name = "rossmann_predictions.csv"

    workflow = nvt.Workflow.load(workflow_path)

    sample_data = cudf.read_csv(data_path, nrows=TEST_N_ROWS)
    sample_data.to_csv(os.path.join(output_dir, workflow_output_test_file_name))
    sample_data_trans = nvt.workflow.workflow._transform_partition(
        sample_data, [workflow.output_node]
    )
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


def _run_movielens_query(client, n_rows):
    workflow_path = os.path.join(os.path.expanduser(MODEL_DIR), "movielens_nvt/1/workflow")
    data_path = os.path.join(
        os.path.expanduser(DATA_DIR), "movielens/test_inference_movielens_data.csv"
    )
    actual_output_filename = os.path.join(
        os.path.expanduser(DATA_DIR), "movielens/movielens_predictions.csv"
    )

    input_col_names = ["movieId", "userId"]
    return _run_query(
        client,
        n_rows,
        "movielens",
        workflow_path,
        data_path,
        actual_output_filename,
        "output",
        input_col_names,
    )


def _run_rossmann_query(client, n_rows):
    workflow_path = os.path.join(os.path.expanduser(MODEL_DIR), "rossmann_nvt/1/workflow")
    data_path = os.path.join(
        os.path.expanduser(DATA_DIR), "rossman/test_inference_rossmann_data.csv"
    )
    actual_output_filename = os.path.join(
        os.path.expanduser(DATA_DIR), "rossman/rossmann_predictions.csv"
    )
    return _run_query(
        client,
        n_rows,
        "rossmann",
        workflow_path,
        data_path,
        actual_output_filename,
        "tf.math.multiply_1",
    )


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
