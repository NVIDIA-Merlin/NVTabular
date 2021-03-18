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
from criteo_parsers import CriteoBenchFastAI
from rossmann_parsers import RossBenchFastAI, RossBenchPytorch, RossBenchTensorFlow

TEST_PATH = dirname(dirname(realpath(__file__)))
DATA_START = os.environ.get("DATASET_DIR", "/raid/criteo/")


def test_criteo_notebook(asv_db, bench_info, tmpdir):
    input_path = os.path.join(DATA_START, "tests/crit_int_pq")
    output_path = os.path.join(DATA_START, "tests/crit_test")
    os.environ["PARTS_PER_CHUNK"] = "1"

    out = _run_notebook(
        tmpdir,
        os.path.join(dirname(TEST_PATH), "examples", "criteo-example.ipynb"),
        input_path,
        output_path,
        # disable rmm.reinitialize, seems to be causing issues
        transform=lambda line: line.replace("rmm.reinitialize(", "# rmm.reinitialize("),
        gpu_id=0,
        batch_size=100000,
    )
    bench_results = CriteoBenchFastAI().get_epochs(out.splitlines())
    bench_results += CriteoBenchFastAI().get_dl_timing(out.splitlines())
    send_results(asv_db, bench_info, bench_results)


@pytest.mark.skip(reason="Criteo-hugectr notebook needs to be updated.")
def test_criteohugectr_notebook(asv_db, bench_info, tmpdir):
    input_path = os.path.join(DATA_START, "criteo/crit_int_pq")
    output_path = os.path.join(DATA_START, "criteo/crit_test")
    os.environ["PARTS_PER_CHUNK"] = "1"

    _run_notebook(
        tmpdir,
        os.path.join(dirname(TEST_PATH), "examples", "hugectr", "criteo-hugectr.ipynb"),
        input_path,
        output_path,
        # disable rmm.reinitialize, seems to be causing issues
        transform=lambda line: line.replace("rmm.reinitialize(", "# rmm.reinitialize("),
        gpu_id="0,1",
        batch_size=100000,
    )


def test_rossman_example(asv_db, bench_info, tmpdir):
    pytest.importorskip("tensorflow")
    data_path = os.path.join(DATA_START, "rossman/data")
    input_path = os.path.join(DATA_START, "rossman/input")
    output_path = os.path.join(DATA_START, "rossman/output")

    notebookpre_path = os.path.join(
        dirname(TEST_PATH), "examples/tabular-data-rossmann", "01-Download-Convert.ipynb"
    )

    out = _run_notebook(tmpdir, notebookpre_path, data_path, input_path, gpu_id=4, clean_up=False)

    notebookpre_path = os.path.join(
        dirname(TEST_PATH), "examples/tabular-data-rossmann", "02-ETL-with-NVTabular.ipynb"
    )

    out = _run_notebook(tmpdir, notebookpre_path, data_path, input_path, gpu_id=4, clean_up=False)

    notebookex_path = os.path.join(
        dirname(TEST_PATH), "examples/tabular-data-rossmann", "04-Training-with-FastAI.ipynb"
    )
    out = _run_notebook(tmpdir, notebookex_path, input_path, output_path, gpu_id=4)
    bench_results = RossBenchFastAI().get_epochs(out.splitlines())
    bench_results += RossBenchFastAI().get_dl_timing(out.splitlines())
    send_results(asv_db, bench_info, bench_results)

    notebookex_path = os.path.join(
        dirname(TEST_PATH), "examples/tabular-data-rossmann", "03b-Training-with-PyTorch.ipynb"
    )
    out = _run_notebook(tmpdir, notebookex_path, input_path, output_path, gpu_id=4)
    bench_results = RossBenchPytorch().get_epochs(out.splitlines())
    bench_results += RossBenchPytorch().get_dl_timing(out.splitlines())
    send_results(asv_db, bench_info, bench_results)

    notebookex_path = os.path.join(
        dirname(TEST_PATH), "examples/tabular-data-rossmann", "03a-Training-with-TF.ipynb"
    )
    out = _run_notebook(tmpdir, notebookex_path, input_path, output_path, gpu_id=4)
    bench_results = RossBenchTensorFlow().get_epochs(out.splitlines())
    bench_results += RossBenchTensorFlow().get_dl_timing(out.splitlines())
    send_results(asv_db, bench_info, bench_results)


def test_tf_inference_training_examples(asv_db, bench_info, tmpdir):

    data_path = DATA_START  # os.path.join(DATA_START, "inference/data/")
    input_path = DATA_START  # os.path.join(DATA_START, "inference/input/")

    notebookpre_path = os.path.join(
        dirname(TEST_PATH), "examples/inference_triton/inference-TF", "movielens-TF.ipynb"
    )

    os.environ["BASE_DIR"] = DATA_START
    os.environ["MODEL_NAME_NVT"] = "movielens_nvt"
    os.environ["MODEL_NAME_TF"] = "movielens_tf"
    os.environ["MODEL_NAME_ENSEMBLE"] = "movielens"
    os.environ["MODEL_PATH"] = os.path.join(DATA_START, "models")

    out = _run_notebook(tmpdir, notebookpre_path, data_path, input_path, gpu_id="0", clean_up=False)

    notebookpre_path = os.path.join(
        dirname(TEST_PATH), "examples/inference_triton/inference-TF", "movielens-multihot-TF.ipynb"
    )

    os.environ["MODEL_NAME_NVT"] = "movielens_nvt_mh"
    os.environ["MODEL_NAME_TF"] = "movielens_tf_mh"
    os.environ["MODEL_NAME_ENSEMBLE"] = "movielens_mh"
    os.environ["MODEL_PATH"] = os.path.join(DATA_START, "models_multihot")

    out = _run_notebook(tmpdir, notebookpre_path, data_path, input_path, gpu_id="0", clean_up=False)


def test_tf_inference_examples(asv_db, bench_info, tmpdir):

    data_path = DATA_START  # os.path.join(DATA_START, "inference/data/")
    input_path = DATA_START  # os.path.join(DATA_START, "inference/input/")

    os.environ["MODEL_BASE_DIR"] = "/model/models/"

    notebookpre_path = os.path.join(
        dirname(TEST_PATH), "examples/inference_triton/inference-TF", "movielens-inference.ipynb"
    )

    out = _run_notebook(tmpdir, notebookpre_path, data_path, input_path, gpu_id="0", clean_up=False)

    os.environ["MODEL_BASE_DIR"] = "/model/models_multihot/"

    notebookpre_path = os.path.join(
        dirname(TEST_PATH),
        "examples/inference_triton/inference-TF",
        "movielens-multihot-inference.ipynb",
    )

    out = _run_notebook(tmpdir, notebookpre_path, data_path, input_path, gpu_id="0", clean_up=False)


def _run_notebook(
    tmpdir,
    notebook_path,
    input_path,
    output_path,
    batch_size=None,
    gpu_id=0,
    clean_up=True,
    transform=None,
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
