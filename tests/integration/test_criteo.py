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
from testbook import testbook

import tests.conftest as test_utils

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
TRITON_SERVER_PATH = find_executable("tritonserver")

INFERENCE_BASE_DIR = os.environ.get("INFERENCE_BASE_DIR", "/tmp/model/")
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
    os.makedirs(output_path)
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


@pytest.mark.skipif(tensorflow is None, reason="tensorflow not installed")
def test_criteo_tf(asv_db, bench_info, tmpdir, report):
    criteo_base(tmpdir)
    output_path = os.path.join(tmpdir, "tests/crit_test")
    # inference_path = os.path.join(output_path, "models")
    # os.makedirs(inference_path)
    os.environ["MODEL_BASE_DIR"] = INFERENCE_MULTI_HOT

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

    notebook = os.path.join(dirname(TEST_PATH), CRITEO_DIR, "04-Triton-Inference-with-TF.ipynb")
    with testbook(
        notebook,
        execute=False,
        timeout=450,
    ) as tb_infer:
        tb_infer.inject(
            f"""
                import os
                os.environ['INPUT_DATA_DIR'] = "{output_path}"
                os.environ['OUTPUT_DATA_DIR'] = "{output_path}"
            """
        )
        tb_infer.execute_cell(list(range(0, 16)))
    with test_utils.run_triton_server(
        INFERENCE_MULTI_HOT,
        "criteo",
        TRITON_SERVER_PATH,
        str(0),
        "tensorflow",
    ):
        with testbook(
            notebook,
            execute=False,
            timeout=450,
        ) as tb_infer:
            tb_infer.inject(
                f"""
                    import os
                    os.environ['INPUT_DATA_DIR'] = "{output_path}"
                    os.environ['OUTPUT_DATA_DIR'] = "{output_path}"
                """
            )
            tb_infer.execute_cell(list(range(19, len(tb_infer.cells))))


@pytest.mark.skipif(hugectr is None, reason="hugectr not installed")
def test_criteo_hugectr(asv_db, bench_info, tmpdir, report):
    criteo_base(tmpdir)
    output_path = os.path.join(tmpdir, "tests/crit_test")

    notebook = os.path.join(dirname(TEST_PATH), CRITEO_DIR, "03-Training-with-HugeCTR.ipynb")

    with testbook(
        notebook,
        execute=False,
        timeout=450,
    ) as tb_train:
        tb_train.inject(
            f"""
                import os
                os.environ['INPUT_DATA_DIR'] = "{output_path}"
                os.environ['OUTPUT_DATA_DIR'] = "{output_path}"
            """
        )
        tb_train.execute_cell(list(range(0, len(tb_train.cells))))

    notebook = os.path.join(
        dirname(TEST_PATH), CRITEO_DIR, "04-Triton-Inference-with-HugeCTR.ipynb"
    )

    with testbook(
        notebook,
        execute=False,
        timeout=450,
    ) as tb_infer:
        tb_infer.inject(
            f"""
                import os
                os.environ['INPUT_DATA_DIR'] = "{output_path}"
                os.environ['OUTPUT_DATA_DIR'] = "{output_path}"
            """
        )
        tb_infer.execute_cell(list(range(0, 16)))
    write_ps_file()
    with test_utils.run_triton_server(
        INFERENCE_BASE_DIR,
        "criteo",
        TRITON_SERVER_PATH,
        str(0),
        "hugectr",
        ps_path="/tmp/model/ps.json",
    ):
        with testbook(
            notebook,
            execute=False,
            timeout=450,
        ) as tb_infer:
            tb_infer.inject(
                f"""
                    import os
                    os.environ['BASE_DIR'] = "{DATA_DIR}"
                    os.environ['INPUT_DATA_DIR'] = "{output_path}"
                    os.environ['OUTPUT_DATA_DIR'] = "{output_path}"
                    import numpy as np
                """
            )
            tb_infer.execute_cell(list(range(18, len(tb_infer.cells))))


# out = _run_notebook(tmpdir, notebook, output_path, output_path, gpu_id="0", clean_up=False)
# notebook = os.path.join(
#     dirname(TEST_PATH), CRITEO_DIR, "04-Triton-Inference-with-HugeCTR.ipynb"
# )
# _run_notebook(tmpdir, notebook, input_path, output_path, gpu_id="0", clean_up=False)


def write_ps_file():
    import json

    config = json.dumps(
        {
            "supportlonglong": "true",
            "models": [
                {
                    "model": "criteo",
                    "sparse_files": ["/tmp/model/criteo/1/0_sparse_9600.model"],
                    "dense_file": "/tmp/model/criteo/1/_dense_9600.model",
                    "network_file": "/tmp/model/criteo/1/criteo.json",
                    "max_batch_size": "64",
                    "gpucache": "true",
                    "hit_rate_threshold": "0.9",
                    "gpucacheper": "0.5",
                    "num_of_worker_buffer_in_pool": "4",
                    "num_of_refresher_buffer_in_pool": "1",
                    "cache_refresh_percentage_per_iteration": 0.2,
                    "deployed_device_list": ["0"],
                    "default_value_for_each_table": ["0.0", "0.0"],
                    "maxnum_catfeature_query_per_table_per_sample": [2, 26],
                    "embedding_vecsize_per_table": [16 for x in range(26)],
                }
            ],
        }
    )

    config = json.loads(config)
    with open("/tmp/model/ps.json", "w", encoding="utf-8") as f:
        json.dump(config, f)
