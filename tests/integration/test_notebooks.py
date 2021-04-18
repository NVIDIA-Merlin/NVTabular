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

import itertools
import json
import os
import shutil
import subprocess
import sys
from os.path import dirname, realpath

import pytest
from benchmark_parsers import send_results
from criteo_parsers import CriteoBenchFastAI, CriteoBenchHugeCTR
from rossmann_parsers import RossBenchFastAI, RossBenchPytorch, RossBenchTensorFlow

TEST_PATH = dirname(dirname(realpath(__file__)))
DATA_START = os.environ.get("DATASET_DIR", "/raid/data/")

INFERENCE_BASE_DIR = "/model/"
INFERENCE_MULTI_HOT = os.path.join(INFERENCE_BASE_DIR, "models/")


def test_criteo_example(asv_db, bench_info, tmpdir):
    input_path = os.path.join(DATA_START, "tests/crit_int_pq")
    output_path = os.path.join(DATA_START, "tests/crit_test")

    notebook_etl = os.path.join(
        dirname(TEST_PATH), "examples/scaling-criteo", "02-ETL-with-NVTabular.ipynb"
    )
    out = _run_notebook(tmpdir, notebook_etl, input_path, output_path, gpu_id="0", clean_up=False, main_block=39)

    # Only run if PyTorch installed
    try:
        import torch

        print(torch.__version__)

        notebook_pytorch = os.path.join(
            dirname(TEST_PATH), "examples/scaling-criteo", "03d-Training-with-FastAI.ipynb"
        )

        out = _run_notebook(
            tmpdir, notebook_pytorch, input_path, output_path, gpu_id="0", clean_up=False
        )

        bench_results = CriteoBenchFastAI().get_epochs(out.splitlines())
        bench_results += CriteoBenchFastAI().get_dl_timing(out.splitlines())
        send_results(asv_db, bench_info, bench_results)
    except ImportError:
        print("Pytorch not installed in this container, skipping 03d-Training-with-FastAI.ipynb")

    # Only run if HugeCTR installed
    try:
        import hugectr

        print(hugectr.__version__)

        notebook_hugectr = os.path.join(
            dirname(TEST_PATH), "examples/scaling-criteo", "03c-Training-with-HugeCTR.ipynb"
        )

        out = _run_notebook(
            tmpdir, notebook_hugectr, input_path, output_path, gpu_id="0", clean_up=False
        )

        bench_results = CriteoBenchHugeCTR().get_epochs(out.splitlines())
        bench_results += CriteoBenchHugeCTR().get_dl_timing(out.splitlines())
        send_results(asv_db, bench_info, bench_results)
    except ImportError:
        print("HugeCTR not installed in this container, skipping 03c-Training-with-HugeCTR.ipynb")


def test_rossman_example(asv_db, bench_info, tmpdir):
    data_path = os.path.join(DATA_START, "rossman/data")
    input_path = os.path.join(DATA_START, "rossman/input")
    output_path = os.path.join(DATA_START, "rossman/output")

    notebookpre_path = os.path.join(
        dirname(TEST_PATH), "examples/tabular-data-rossmann", "01-Download-Convert.ipynb"
    )

    out = _run_notebook(tmpdir, notebookpre_path, data_path, input_path, gpu_id="4", clean_up=False)

    notebookpre_path = os.path.join(
        dirname(TEST_PATH), "examples/tabular-data-rossmann", "02-ETL-with-NVTabular.ipynb"
    )

    out = _run_notebook(tmpdir, notebookpre_path, data_path, input_path, gpu_id="4", clean_up=False)

    # Only run if PyTorch installed
    try:
        import torch

        print(torch.__version__)

        notebookex_path = os.path.join(
            dirname(TEST_PATH), "examples/tabular-data-rossmann", "04-Training-with-FastAI.ipynb"
        )
        out = _run_notebook(tmpdir, notebookex_path, input_path, output_path, gpu_id="4")
        bench_results = RossBenchFastAI().get_epochs(out.splitlines())
        bench_results += RossBenchFastAI().get_dl_timing(out.splitlines())
        send_results(asv_db, bench_info, bench_results)

        notebookex_path = os.path.join(
            dirname(TEST_PATH), "examples/tabular-data-rossmann", "03b-Training-with-PyTorch.ipynb"
        )
        out = _run_notebook(tmpdir, notebookex_path, input_path, output_path, gpu_id="4")
        bench_results = RossBenchPytorch().get_epochs(out.splitlines())
        bench_results += RossBenchPytorch().get_dl_timing(out.splitlines())
        send_results(asv_db, bench_info, bench_results)

    except ImportError:
        print("Pytorch not installed in this container, skipping tests")

    # Only run if TensorFlow installed
    try:
        import tensorflow

        print(tensorflow.__version__)

        notebookex_path = os.path.join(
            dirname(TEST_PATH), "examples/tabular-data-rossmann", "03a-Training-with-TF.ipynb"
        )
        out = _run_notebook(tmpdir, notebookex_path, input_path, output_path, gpu_id="4")
        bench_results = RossBenchTensorFlow().get_epochs(out.splitlines())
        bench_results += RossBenchTensorFlow().get_dl_timing(out.splitlines())
        send_results(asv_db, bench_info, bench_results)
    except ImportError:
        print("TensorFlow not installed in this container, skipping 03a-Training-with-TF.ipynb")


def test_tf_inference_training_examples(asv_db, bench_info, tmpdir):
    # Tensorflow required to run this test
    pytest.importorskip("tensorflow")
    data_path = os.path.join(INFERENCE_BASE_DIR, "data/")
    input_path = os.path.join(INFERENCE_BASE_DIR, "data/")
    
    os.environ["BASE_DIR"] = INFERENCE_BASE_DIR
    os.environ["MODEL_NAME_NVT"] = "movielens_nvt"
    os.environ["MODEL_NAME_TF"] = "movielens_tf"
    os.environ["MODEL_NAME_ENSEMBLE"] = "movielens"
    os.environ["MODEL_BASE_DIR"] = INFERENCE_MULTI_HOT

    notebookpre_path = os.path.join(
        dirname(TEST_PATH), "examples/getting-started-movielens", "01-Download-Convert.ipynb"
    )
    _run_notebook(tmpdir, notebookpre_path, data_path, input_path, gpu_id="0", clean_up=False)
    
    notebookpre_path = os.path.join(
        dirname(TEST_PATH), "examples/getting-started-movielens", "02-ETL-with-NVTabular.ipynb"
    )
    _run_notebook(tmpdir, notebookpre_path, data_path, input_path, gpu_id="0", clean_up=False)
    
    notebookpre_path = os.path.join(
        dirname(TEST_PATH), "examples/getting-started-movielens", "03a-Training-with-TF.ipynb"
    )
    _run_notebook(tmpdir, notebookpre_path, data_path, input_path, gpu_id="0", clean_up=False)


def test_tf_inference_multihot_examples(asv_db, bench_info, tmpdir):
    # Tritonclient required for this test
    pytest.importorskip("tritonclient")

    data_path = os.path.join(INFERENCE_BASE_DIR, "data/")
    input_path = os.path.join(INFERENCE_BASE_DIR, "data/")

    os.environ["MODEL_BASE_DIR"] = INFERENCE_MULTI_HOT

    notebookpre_path = os.path.join(
        dirname(TEST_PATH), "examples/getting-started-movielens", "04a-Triton-Inference-with-TF.ipynb",
    )

    _run_notebook(tmpdir, notebookpre_path, data_path, input_path, gpu_id="0", clean_up=True)


def _run_notebook(
    tmpdir,
    notebook_path,
    input_path,
    output_path,
    batch_size=None,
    gpu_id=0,
    clean_up=True,
    transform=None,
    main_block=-1,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU_TARGET_ID", gpu_id)

    if not os.path.exists(input_path):
        os.makedirs(input_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if batch_size:
        os.environ["BATCH_SIZE"] = os.environ.get("BATCH_SIZE", batch_size)

    os.environ["INPUT_DATA_DIR"] = input_path
    os.environ["OUTPUT_DATA_DIR"] = output_path
    # read in the notebook as JSON, and extract a python script from it
    notebook = json.load(open(notebook_path))
    source_cells = [cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"]
    
    lines = [
        transform(line.rstrip()) if transform else line
        for line in itertools.chain(*source_cells)
        if not (line.startswith("%") or line.startswith("!"))
    ]

    # Add guarding block and indentation
    if main_block >= 0:
        lines.insert(main_block, "if __name__ == \"__main__\":")
        for i in range(main_block+1, len(lines)):
            lines[i] = "    " + lines[i]
            
    # save the script to a file, and run with the current python executable
    # we're doing this in a subprocess to avoid some issues using 'exec'
    # that were causing a segfault with globals of the exec'ed function going
    # out of scope
    script_path = os.path.join(tmpdir, "notebook.py")
    with open(script_path, "w") as script:
        script.write("\n".join(lines))
    output = subprocess.check_output([sys.executable, script_path])
    # save location will default to run location
    output = output.decode("utf-8")
    _, note_name = os.path.split(notebook_path)
    note_name = note_name.split(".")[0]
    if output:
        with open(f"test_res_{note_name}", "w+") as w_file:
            w_file.write(output)
    # clear out products
    if clean_up:
        shutil.rmtree(output_path)
    return output
