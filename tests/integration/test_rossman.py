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
from os.path import dirname, realpath

import pytest
from testbook import testbook

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

INFERENCE_BASE_DIR = "/model/"
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
    data_path = os.path.join(DATA_DIR, "rossman/data")
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
                os.environ['INPUT_DATA_DIR'] = "{data_path}"
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


@pytest.mark.skipif(torch is None, reason="pytorch not installed")
def test_rossman_torch(asv_db, bench_info, tmpdir, devices, report):
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
