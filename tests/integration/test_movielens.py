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

import os
from distutils.spawn import find_executable
from os.path import dirname, realpath

import pytest
from common.utils import _run_query
from testbook import testbook

import tests.conftest as test_utils

try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow
except ImportError:
    tensorflow = None

DATA_DIR = os.environ.get("DATASET_DIR", "/raid/data/")
TEST_PATH = dirname(dirname(realpath(__file__)))

INFERENCE_BASE_DIR = "/tmp/model/"
INFERENCE_MULTI_HOT = os.path.join(INFERENCE_BASE_DIR, "models/")

CRITEO_DIR = "examples/scaling-criteo"
ROSSMAN_DIR = "examples/tabular-data-rossmann"
MOVIELENS_DIR = "examples/getting-started-movielens"
TRITON_SERVER_PATH = find_executable("tritonserver")


allowed_hosts = [
    "merlin-hugectr",
    "merlin-tensorflow",
    "merlin-pytorch",
]


def movielens_base(tmpdir):
    data_path = os.path.join(DATA_DIR, "movielens-25m")
    input_path = os.path.join(tmpdir, "movielens/input")
    os.makedirs(input_path)
    os.environ["BASE_DIR"] = INFERENCE_BASE_DIR
    os.environ["MODEL_NAME_NVT"] = "movielens_nvt"
    os.environ["MODEL_NAME_TF"] = "movielens_tf"
    os.environ["MODEL_NAME_ENSEMBLE"] = "movielens"
    os.environ["MODEL_BASE_DIR"] = INFERENCE_MULTI_HOT
    os.environ["MODEL_PATH"] = INFERENCE_MULTI_HOT

    # Run Tensorflow or PyTorch
    # Run Download & Convert for all
    notebook = os.path.join(dirname(TEST_PATH), MOVIELENS_DIR, "01-Download-Convert.ipynb")

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

    # _run_notebook(tmpdir, notebook, data_path, input_path, gpu_id=devices, clean_up=False)

    # Run ETL for all
    notebook = os.path.join(dirname(TEST_PATH), MOVIELENS_DIR, "02-ETL-with-NVTabular.ipynb")
    with testbook(
        notebook,
        execute=False,
        timeout=450,
    ) as tb_nvt:
        tb_nvt.inject(
            f"""
                import os
                os.environ['INPUT_DATA_DIR'] = "{data_path}"
                os.environ['OUTPUT_DATA_DIR'] = "{input_path}"
            """
        )
        tb_nvt.execute_cell(list(range(0, len(tb_nvt.cells))))


@pytest.mark.skipif(tensorflow is None, reason="tensorflow not installed")
def test_movielens_tf(asv_db, bench_info, tmpdir, devices):
    movielens_base(tmpdir)

    data_path = os.path.join(DATA_DIR, "movielens-25m")
    input_path = os.path.join(tmpdir, "movielens/input")
    os.environ["BASE_DIR"] = INFERENCE_BASE_DIR
    os.environ["MODEL_NAME_NVT"] = "movielens_nvt"
    os.environ["MODEL_NAME_TF"] = "movielens_tf"
    os.environ["MODEL_NAME_ENSEMBLE"] = "movielens"
    os.environ["MODEL_BASE_DIR"] = INFERENCE_MULTI_HOT
    os.environ["MODEL_PATH"] = INFERENCE_MULTI_HOT

    notebook = os.path.join(dirname(TEST_PATH), MOVIELENS_DIR, "03-Training-with-TF.ipynb")
    with testbook(
        notebook,
        execute=False,
        timeout=450,
    ) as tb_train_tf:
        tb_train_tf.inject(
            f"""
                import os
                os.environ['INPUT_DATA_DIR'] = "{data_path}"
                os.environ['OUTPUT_DATA_DIR'] = "{input_path}"
            """
        )
        tb_train_tf.execute_cell(list(range(0, len(tb_train_tf.cells))))
    create_movielens_inference_data(INFERENCE_MULTI_HOT, DATA_DIR, input_path, 100)
    with test_utils.run_triton_server(
        INFERENCE_MULTI_HOT,
        "movielens",
        TRITON_SERVER_PATH,
        str(0),
        "tensorflow",
    ) as client:
        diff, run_time = _run_movielens_query(client, 3, INFERENCE_MULTI_HOT, input_path)

        assert (diff < 0.00001).all()


@pytest.mark.skipif(torch is None, reason="pytorch not installed")
def test_movielens_torch(asv_db, bench_info, tmpdir, devices):
    movielens_base(tmpdir)

    data_path = os.path.join(DATA_DIR, "movielens-25m")
    input_path = os.path.join(tmpdir, "movielens/input")
    os.environ["BASE_DIR"] = INFERENCE_BASE_DIR
    os.environ["MODEL_NAME_NVT"] = "movielens_nvt"
    os.environ["MODEL_NAME_TF"] = "movielens_tf"
    os.environ["MODEL_NAME_ENSEMBLE"] = "movielens"
    os.environ["MODEL_BASE_DIR"] = INFERENCE_MULTI_HOT
    os.environ["MODEL_PATH"] = INFERENCE_MULTI_HOT

    # _run_notebook(tmpdir, notebook, data_path, input_path, gpu_id=devices, clean_up=False)
    notebook = os.path.join(dirname(TEST_PATH), MOVIELENS_DIR, "03-Training-with-PyTorch.ipynb")

    with testbook(
        notebook,
        execute=False,
        timeout=450,
    ) as tb_train_torch:
        tb_train_torch.inject(
            f"""
                import os
                os.environ['INPUT_DATA_DIR'] = "{data_path}"
                os.environ['OUTPUT_DATA_DIR'] = "{input_path}"
            """
        )
        tb_train_torch.execute_cell(list(range(0, len(tb_train_torch.cells))))


def create_movielens_inference_data(model_dir, data_dir, output_dir, nrows):
    import glob

    import cudf
    from tensorflow import keras

    import nvtabular as nvt
    from nvtabular.loader.tensorflow import KerasSequenceLoader

    workflow_path = os.path.join(os.path.expanduser(model_dir), "movielens_nvt/1/workflow")
    model_path = os.path.join(os.path.expanduser(model_dir), "movielens_tf/1/model.savedmodel")
    data_path = os.path.join(os.path.expanduser(data_dir), "movielens/data/valid.parquet")
    output_dir = os.path.join(os.path.expanduser(output_dir), "movielens/")
    os.makedirs(output_dir)
    workflow_output_test_file_name = "test_inference_movielens_data.csv"
    workflow_output_test_trans_file_name = "test_inference_movielens_data_trans.parquet"
    prediction_file_name = "movielens_predictions.csv"

    workflow = nvt.Workflow.load(workflow_path)

    sample_data = cudf.read_parquet(data_path, nrows=nrows)
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
        batch_size=nrows,
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


def _run_movielens_query(client, n_rows, model_dir, output_dir):
    workflow_path = os.path.join(os.path.expanduser(model_dir), "movielens_nvt/1/workflow")
    data_path = os.path.join(
        os.path.expanduser(output_dir), "movielens/test_inference_movielens_data.csv"
    )
    actual_output_filename = os.path.join(
        os.path.expanduser(output_dir), "movielens/movielens_predictions.csv"
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
