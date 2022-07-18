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
import os
from distutils.spawn import find_executable
from os.path import dirname, realpath

import cudf
import pytest
from common.utils import _run_query
from testbook import testbook

import nvtabular as nvt
import tests.conftest as test_utils

try:
    import torch
except ImportError:
    torch = None

try:
    import fastai
except ImportError:
    fastai = None

try:
    import tensorflow
except ImportError:
    tensorflow = None

DATA_DIR = os.environ.get("DATASET_DIR", "/raid/data/")
TEST_PATH = dirname(dirname(realpath(__file__)))
TRITON_SERVER_PATH = find_executable("tritonserver")

INFERENCE_BASE_DIR = "/tmp/model/"
INFERENCE_MULTI_HOT = os.path.join(INFERENCE_BASE_DIR, "models/")

CRITEO_DIR = "examples/scaling-criteo"
ROSSMAN_DIR = "examples/tabular-data-rossmann"
MOVIELENS_DIR = "examples/getting-started-movielens"

allowed_hosts = [
    "merlin-hugectr",
    "merlin-tensorflow",
    "merlin-pytorch",
]


def rossman_base(tmpdir):
    data_path = os.path.join(DATA_DIR, "rossman/")
    input_path = os.path.join(tmpdir, "rossman/input")
    os.makedirs(input_path)

    # Run Downloa   d & Convert for all
    notebook = os.path.join(dirname(TEST_PATH), ROSSMAN_DIR, "01-Download-Convert.ipynb")
    with testbook(
        notebook,
        execute=False,
        timeout=450,
    ) as tb_dl_convert:
        tb_dl_convert.inject(
            f"""
                import os
                os.environ['INPUT_DATA_DIR'] = "{data_path}"
                os.environ['OUTPUT_DATA_DIR'] = "{input_path}"
            """
        )
        tb_dl_convert.execute_cell(list(range(0, len(tb_dl_convert.cells))))
    # out = _run_notebook(tmpdir, notebook, data_path, input_path, gpu_id=devices, clean_up=False)

    # Run ETL for all
    notebook = os.path.join(dirname(TEST_PATH), ROSSMAN_DIR, "02-ETL-with-NVTabular.ipynb")
    with testbook(
        notebook,
        execute=False,
        timeout=450,
    ) as tb_nvt:
        tb_nvt.inject(
            f"""
                import os
                os.environ['INPUT_DATA_DIR'] = "{input_path}"
                os.environ['OUTPUT_DATA_DIR'] = "{input_path}"
            """
        )
        tb_nvt.execute_cell(list(range(0, len(tb_nvt.cells))))


@pytest.mark.skipif(tensorflow is None, reason="tensorflow not installed")
def test_rossman_tf(asv_db, bench_info, tmpdir, devices, report):
    rossman_base(tmpdir)
    input_path = os.path.join(tmpdir, "rossman/input")
    output_path = os.path.join(tmpdir, "rossman/output")
    os.makedirs(output_path)

    os.environ["BASE_DIR"] = INFERENCE_BASE_DIR

    # Run training for PyTorch container
    notebook = os.path.join(dirname(TEST_PATH), ROSSMAN_DIR, "03-Training-with-TF.ipynb")

    with testbook(
        notebook,
        execute=False,
        timeout=450,
    ) as tb_training:
        tb_training.inject(
            f"""
                import os
                os.environ['INPUT_DATA_DIR'] = "{input_path}"
                os.environ['OUTPUT_DATA_DIR'] = "{output_path}"
            """
        )
        tb_training.execute_cell(list(range(0, len(tb_training.cells))))
    create_rossman_inference_data(INFERENCE_MULTI_HOT, DATA_DIR, output_path, 100)
    with test_utils.run_triton_server(
        INFERENCE_MULTI_HOT,
        "rossmann",
        TRITON_SERVER_PATH,
        str(0),
        "tensorflow",
    ) as client:
        diff, run_time = _run_rossmann_query(client, 3, INFERENCE_MULTI_HOT, output_path)

        assert (diff < 0.00001).all()


@pytest.mark.skipif(torch is None, reason="pytorch not installed")
def test_rossman_torch(asv_db, bench_info, tmpdir, devices, report):
    rossman_base(tmpdir)
    input_path = os.path.join(tmpdir, "rossman/input")
    output_path = os.path.join(tmpdir, "rossman/output")
    os.makedirs(output_path)

    os.environ["BASE_DIR"] = INFERENCE_BASE_DIR

    # Run training for PyTorch container
    notebook = os.path.join(dirname(TEST_PATH), ROSSMAN_DIR, "03-Training-with-PyTorch.ipynb")

    with testbook(
        notebook,
        execute=False,
        timeout=450,
    ) as tb_training:
        tb_training.inject(
            f"""
                import os
                os.environ['INPUT_DATA_DIR'] = "{input_path}"
                os.environ['OUTPUT_DATA_DIR'] = "{input_path}"
            """
        )
        tb_training.execute_cell(list(range(0, len(tb_training.cells))))


@pytest.mark.skipif(torch is None or fastai is None, reason="pytorch & fastai not installed")
def test_rossman_fastai(asv_db, bench_info, tmpdir, devices, report):
    rossman_base(tmpdir)
    input_path = os.path.join(tmpdir, "rossman/input")
    output_path = os.path.join(tmpdir, "rossman/output")
    os.makedirs(output_path)

    os.environ["BASE_DIR"] = INFERENCE_BASE_DIR

    # Run training for PyTorch container
    notebook = os.path.join(dirname(TEST_PATH), ROSSMAN_DIR, "03-Training-with-FastAI.ipynb")

    with testbook(
        notebook,
        execute=False,
        timeout=450,
    ) as tb_training:
        tb_training.inject(
            f"""
                import os
                os.environ['INPUT_DATA_DIR'] = "{input_path}"
                os.environ['OUTPUT_DATA_DIR'] = "{input_path}"
            """
        )
        tb_training.execute_cell(list(range(0, len(tb_training.cells))))


def create_rossman_inference_data(model_dir, data_dir, output_dir, nrows):
    import tensorflow as tf
    from tensorflow import keras

    from nvtabular.loader.tensorflow import KerasSequenceLoader

    workflow_path = os.path.join(os.path.expanduser(model_dir), "rossmann_nvt/1/workflow")
    model_path = os.path.join(os.path.expanduser(model_dir), "rossmann_tf/1/model.savedmodel")
    data_path = os.path.join(os.path.expanduser(data_dir), "rossman/input/valid.csv")
    output_dir = os.path.join(os.path.expanduser(output_dir), "rossman/")
    os.makedirs(output_dir)
    workflow_output_test_file_name = "test_inference_rossmann_data.csv"
    workflow_output_test_trans_file_name = "test_inference_rossmann_data_trans.parquet"
    prediction_file_name = "rossmann_predictions.csv"

    workflow = nvt.Workflow.load(workflow_path)

    sample_data = cudf.read_csv(data_path, nrows=nrows)
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
        batch_size=nrows,
        label_names=[],
        shuffle=False,
        buffer_size=0.06,  # amount of data, as a fraction of GPU memory, to load at once
    )

    tf_model = keras.models.load_model(model_path, custom_objects={"rmspe_tf": rmspe_tf})

    pred = tf_model.predict(train_dataset)
    cudf_pred = cudf.DataFrame(pred)
    cudf_pred.to_csv(os.path.join(output_dir, prediction_file_name))

    os.remove(os.path.join(output_dir, workflow_output_test_trans_file_name))


def _run_rossmann_query(client, n_rows, model_dir, output_dir):
    workflow_path = os.path.join(os.path.expanduser(model_dir), "rossmann_nvt/1/workflow")
    data_path = os.path.join(
        os.path.expanduser(output_dir), "rossman/test_inference_rossmann_data.csv"
    )
    actual_output_filename = os.path.join(
        os.path.expanduser(output_dir), "rossman/rossmann_predictions.csv"
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


def _make_categorical_embedding_column(name, dictionary_size, embedding_dim):
    import tensorflow as tf

    return tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity(name, dictionary_size), embedding_dim
    )


def rmspe_tf(y_true, y_pred):
    import tensorflow as tf

    y_true = tf.exp(y_true) - 1
    y_pred = tf.exp(y_pred) - 1

    percent_error = (y_true - y_pred) / y_true
    return tf.sqrt(tf.reduce_mean(percent_error**2))
