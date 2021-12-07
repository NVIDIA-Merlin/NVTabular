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
from common.parsers.benchmark_parsers import send_results
from common.parsers.criteo_parsers import CriteoBenchFastAI, CriteoBenchHugeCTR
from common.parsers.rossmann_parsers import RossBenchFastAI, RossBenchPytorch, RossBenchTensorFlow
from common.utils import _run_notebook

DATA_DIR = os.environ.get("DATASET_DIR", "/raid/data/")
TEST_PATH = dirname(dirname(realpath(__file__)))

INFERENCE_BASE_DIR = "/model/"
INFERENCE_MULTI_HOT = os.path.join(INFERENCE_BASE_DIR, "models/")

CRITEO_DIR = "examples/scaling-criteo"
ROSSMAN_DIR = "examples/tabular-data-rossmann"
MOVIELENS_DIR = "examples/getting-started-movielens"


def test_criteo(asv_db, bench_info, tmpdir):
    input_path = os.path.join(DATA_DIR, "tests/crit_int_pq")
    output_path = os.path.join(DATA_DIR, "tests/crit_test")

    # Run ETL for all containerss
    notebook = os.path.join(dirname(TEST_PATH), CRITEO_DIR, "02-ETL-with-NVTabular.ipynb")
    out = _run_notebook(
        tmpdir,
        notebook,
        input_path,
        output_path,
        gpu_id="0",
        clean_up=False,
        params=[0.4, 0.5, 0.1],
        main_block=39,
    )

    # Run training for PyTorch container
    try:
        notebook = os.path.join(dirname(TEST_PATH), CRITEO_DIR, "03-Training-with-FastAI.ipynb")
        import torch

        print(torch.__version__)
        out = _run_notebook(tmpdir, notebook, input_path, output_path, gpu_id="0", clean_up=False)
        bench_results = CriteoBenchFastAI().get_epochs(out.splitlines())
        bench_results += CriteoBenchFastAI().get_dl_timing(out.splitlines())
        send_results(asv_db, bench_info, bench_results)
    except ImportError:
        print("Pytorch not installed, skipping " + notebook)

    # Run training for HugeCTR container
    try:
        notebook = os.path.join(dirname(TEST_PATH), CRITEO_DIR, "03-Training-with-HugeCTR.ipynb")
        import hugectr

        print(hugectr.__version__)
        out = _run_notebook(tmpdir, notebook, input_path, output_path, gpu_id="0", clean_up=False)
        bench_results = CriteoBenchHugeCTR().get_epochs(out.splitlines())
        bench_results += CriteoBenchHugeCTR().get_dl_timing(out.splitlines())
        send_results(asv_db, bench_info, bench_results)
    except ImportError:
        print("HugeCTR not installed, skipping " + notebook)

    # Run training for TensorFlow container
    try:
        notebook = os.path.join(dirname(TEST_PATH), CRITEO_DIR, "03-Training-with-TF.ipynb")
        import tensorflow

        print(tensorflow.__version__)
        out = _run_notebook(tmpdir, notebook, input_path, output_path, gpu_id="0", clean_up=False)
    except ImportError:
        print("Tensorflow not installed, skipping " + notebook)


def test_rossman(asv_db, bench_info, tmpdir, devices):
    data_path = os.path.join(DATA_DIR, "rossman/data")
    input_path = os.path.join(DATA_DIR, "rossman/input")
    output_path = os.path.join(DATA_DIR, "rossman/output")

    # Run Download & Convert for all
    notebook = os.path.join(dirname(TEST_PATH), ROSSMAN_DIR, "01-Download-Convert.ipynb")
    out = _run_notebook(tmpdir, notebook, data_path, input_path, gpu_id=devices, clean_up=False)

    # Run ETL for all
    notebook = os.path.join(dirname(TEST_PATH), ROSSMAN_DIR, "02-ETL-with-NVTabular.ipynb")
    out = _run_notebook(tmpdir, notebook, data_path, input_path, gpu_id=devices, clean_up=False)

    os.environ["BASE_DIR"] = INFERENCE_BASE_DIR

    # Run training for PyTorch container
    try:
        notebook = os.path.join(dirname(TEST_PATH), ROSSMAN_DIR, "03-Training-with-FastAI.ipynb")
        import torch

        print(torch.__version__)
        out = _run_notebook(
            tmpdir, notebook, input_path, input_path, gpu_id=devices, clean_up=False
        )
        bench_results = RossBenchFastAI().get_epochs(out.splitlines())
        bench_results += RossBenchFastAI().get_dl_timing(out.splitlines())
        send_results(asv_db, bench_info, bench_results)

        notebook = os.path.join(dirname(TEST_PATH), ROSSMAN_DIR, "03-Training-with-PyTorch.ipynb")
        out = _run_notebook(tmpdir, notebook, input_path, input_path, gpu_id=devices)
        bench_results = RossBenchPytorch().get_epochs(out.splitlines())
        bench_results += RossBenchPytorch().get_dl_timing(out.splitlines())
        send_results(asv_db, bench_info, bench_results)
    except ImportError:
        print("PyTorch not installed, skipping " + notebook)

    # Run training for TensorFlow container
    try:
        notebook = os.path.join(dirname(TEST_PATH), ROSSMAN_DIR, "03-Training-with-TF.ipynb")
        import tensorflow

        print(tensorflow.__version__)
        out = _run_notebook(
            tmpdir, notebook, input_path, output_path, gpu_id=devices, clean_up=False
        )
        bench_results = RossBenchTensorFlow().get_epochs(out.splitlines())
        bench_results += RossBenchTensorFlow().get_dl_timing(out.splitlines())
        send_results(asv_db, bench_info, bench_results)
    except ImportError:
        print("Tensorflow not installed, skipping " + notebook)


def test_movielens(asv_db, bench_info, tmpdir, devices):
    data_path = os.path.join(DATA_DIR, "movielens/data")
    input_path = os.path.join(DATA_DIR, "movielens/input")

    os.environ["BASE_DIR"] = INFERENCE_BASE_DIR
    os.environ["MODEL_NAME_NVT"] = "movielens_nvt"
    os.environ["MODEL_NAME_TF"] = "movielens_tf"
    os.environ["MODEL_NAME_ENSEMBLE"] = "movielens"
    os.environ["MODEL_BASE_DIR"] = INFERENCE_MULTI_HOT
    os.environ["MODEL_PATH"] = INFERENCE_MULTI_HOT

    # Run Tensorflow or PyTorch
    # Run Download & Convert for all
    notebook = os.path.join(dirname(TEST_PATH), MOVIELENS_DIR, "01-Download-Convert.ipynb")
    _run_notebook(tmpdir, notebook, data_path, input_path, gpu_id=devices, clean_up=False)

    # Run ETL for all
    notebook = os.path.join(dirname(TEST_PATH), MOVIELENS_DIR, "02-ETL-with-NVTabular.ipynb")
    _run_notebook(tmpdir, notebook, data_path, input_path, gpu_id=devices, clean_up=False)

    # Run training for PyTorch container
    try:
        notebook = os.path.join(dirname(TEST_PATH), MOVIELENS_DIR, "03-Training-with-PyTorch.ipynb")
        import torch

        print(torch.__version__)
        _run_notebook(tmpdir, notebook, data_path, input_path, gpu_id=devices, clean_up=False)
    except ImportError:
        print("Pytorch not installed, skipping " + notebook)

    # Run training for TensorFlow container
    try:
        notebook = os.path.join(dirname(TEST_PATH), MOVIELENS_DIR, "03-Training-with-TF.ipynb")
        import tensorflow

        print(tensorflow.__version__)
        _run_notebook(tmpdir, notebook, data_path, input_path, gpu_id=devices, clean_up=False)
    except ImportError:
        print("Tensorflow not installed, skipping " + notebook)


def test_inference(asv_db, bench_info, tmpdir, devices):
    # Tritonclient required for this test
    pytest.importorskip("tritonclient")
    # data_path = os.path.join(INFERENCE_BASE_DIR, "data/")
    input_path = os.path.join(INFERENCE_BASE_DIR, "data/")
    output_path = os.path.join(INFERENCE_BASE_DIR, "data/output")

    os.environ["MODEL_BASE_DIR"] = INFERENCE_MULTI_HOT

    # Run Criteo inference
    notebook = os.path.join(
        dirname(TEST_PATH), CRITEO_DIR, "04-Triton-Inference-with-HugeCTR.ipynb"
    )
    _run_notebook(tmpdir, notebook, input_path, output_path, gpu_id=devices, clean_up=False)

    notebook = os.path.join(dirname(TEST_PATH), CRITEO_DIR, "04-Triton-Inference-with-TF.ipynb")
    _run_notebook(tmpdir, notebook, input_path, output_path, gpu_id=devices, clean_up=False)

    # Run Movielens inference
    notebook = os.path.join(
        dirname(TEST_PATH), MOVIELENS_DIR, "inference-HugeCTR/Triton-Inference-with-HugeCTR.ipynb"
    )
    _run_notebook(tmpdir, notebook, input_path, output_path, gpu_id=devices, clean_up=False)

    notebook = os.path.join(dirname(TEST_PATH), MOVIELENS_DIR, "04-Triton-Inference-with-TF.ipynb")
    _run_notebook(tmpdir, notebook, input_path, output_path, gpu_id=devices, clean_up=False)
