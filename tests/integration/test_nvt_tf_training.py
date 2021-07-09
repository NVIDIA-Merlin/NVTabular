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

# External dependencies
import os

import cudf
import pytest
import tensorflow as tf
from tensorflow import keras

import nvtabular as nvt
from nvtabular.loader.tensorflow import KerasSequenceLoader

TEST_N_ROWS = 64
MODEL_DIR = "~/nvt-examples/models/"
DATA_DIR = "~/nvt-examples/data/"


@pytest.mark.skipif(TEST_N_ROWS is None, reason="Requires TEST_N_ROWS")
@pytest.mark.skipif(MODEL_DIR is None, reason="Requires MODEL_DIR")
@pytest.mark.skipif(DATA_DIR is None, reason="Requires DATA_DIR")
def test_nvt_tf_rossmann_training():
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
    # map back into "true" space by undoing transform
    y_true = tf.exp(y_true) - 1
    y_pred = tf.exp(y_pred) - 1

    percent_error = (y_true - y_pred) / y_true
    return tf.sqrt(tf.reduce_mean(percent_error ** 2))


def _make_categorical_embedding_column(name, dictionary_size, embedding_dim):
    return tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity(name, dictionary_size), embedding_dim
    )
