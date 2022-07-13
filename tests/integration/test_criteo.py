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
    import fastai
except ImportError:
    fastai = None

try:
    import hugectr
except ImportError:
    hugectr = None

try:
    import tensorflow
except ImportError:
    tensorflow = None

try:
    import torch
except ImportError:
    torch = None


DATA_DIR = os.environ.get("DATASET_DIR", "/raid/data/criteo")
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


def criteo_base(tmpdir):
    input_path = os.path.join(DATA_DIR, "tests/crit_int_pq")
    output_path = os.path.join(tmpdir, "tests/crit_test")

    # Run ETL for all containerss
    notebook = os.path.join(dirname(TEST_PATH), CRITEO_DIR, "02-ETL-with-NVTabular.ipynb")
    with testbook(
        notebook,
        execute=False,
        timeout=450,
    ) as tb_dl_convert:
        tb_dl_convert.inject(
            f"""
                import os
                os.environ['INPUT_DATA_DIR'] = "{input_path}"
                os.environ['OUTPUT_DATA_DIR'] = "{output_path}"
            """
        )
        tb_dl_convert.execute_cell(list(range(0, len(tb_dl_convert.cells))))


@pytest.mark.skipif(tensorflow is None, reason="tensorflow not installed")
def test_criteo_tf(asv_db, bench_info, tmpdir, report):
    criteo_base(tmpdir)
    output_path = os.path.join(tmpdir, "tests/crit_test")

    notebook = os.path.join(dirname(TEST_PATH), CRITEO_DIR, "03-Training-with-TF.ipynb")
    with testbook(
        notebook,
        execute=False,
        timeout=450,
    ) as tb_train_torch:
        tb_train_torch.inject(
            f"""
                import os
                os.environ['INPUT_DATA_DIR'] = "{output_path}"
                os.environ['OUTPUT_DATA_DIR'] = "{output_path}"
            """
        )
        tb_train_torch.execute_cell(list(range(0, len(tb_train_torch.cells))))


@pytest.mark.skipif(torch is None or fastai is None, reason="pytorch & fastai not installed")
def test_criteo_fastai(asv_db, bench_info, tmpdir, report):
    criteo_base(tmpdir)
    output_path = os.path.join(tmpdir, "tests/crit_test")

    notebook = os.path.join(dirname(TEST_PATH), CRITEO_DIR, "03-Training-with-FastAI.ipynb")
    with testbook(
        notebook,
        execute=False,
        timeout=450,
    ) as tb_train_torch:
        tb_train_torch.inject(
            f"""
                import os
                os.environ['INPUT_DATA_DIR'] = "{output_path}"
                os.environ['OUTPUT_DATA_DIR'] = "{output_path}"
            """
        )
        tb_train_torch.execute_cell(list(range(0, len(tb_train_torch.cells))))


@pytest.mark.skipif(hugectr is None, reason="hugectr not installed")
def test_criteo_hugectr(asv_db, bench_info, tmpdir, report):
    criteo_base(tmpdir)
    output_path = os.path.join(tmpdir, "tests/crit_test")

    notebook = os.path.join(dirname(TEST_PATH), CRITEO_DIR, "03-Training-with-HugeCTR.ipynb")

    with testbook(
        notebook,
        execute=False,
        timeout=450,
    ) as tb_train_torch:
        tb_train_torch.inject(
            f"""
                import os
                os.environ['INPUT_DATA_DIR'] = "{output_path}"
                os.environ['OUTPUT_DATA_DIR'] = "{output_path}"
            """
        )
        tb_train_torch.execute_cell(list(range(0, len(tb_train_torch.cells))))


# out = _run_notebook(tmpdir, notebook, output_path, output_path, gpu_id="0", clean_up=False)
# notebook = os.path.join(
#     dirname(TEST_PATH), CRITEO_DIR, "04-Triton-Inference-with-HugeCTR.ipynb"
# )
# _run_notebook(tmpdir, notebook, input_path, output_path, gpu_id="0", clean_up=False)
