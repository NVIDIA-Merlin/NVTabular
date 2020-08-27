import itertools
import json
import os
import shutil
import subprocess
import sys
from os.path import dirname, realpath

import pytest

TEST_PATH = dirname(dirname(realpath(__file__)))
DATA_START = os.environ.get("DATASET_DIR", "/raid/data")


def test_criteo_notebook(tmpdir):
    input_path = os.path.join(DATA_START, "criteo/crit_int_pq")
    output_path = os.path.join(DATA_START, "criteo/crit_test")
    os.environ["PARTS_PER_CHUNK"] = "1"

    _run_notebook(
        tmpdir,
        os.path.join(dirname(TEST_PATH), "examples", "criteo-example.ipynb"),
        input_path,
        output_path,
        # disable rmm.reinitialize, seems to be causing issues
        transform=lambda line: line.replace("rmm.reinitialize(", "# rmm.reinitialize("),
        gpu_id=3,
        batch_size=100000,
    )


def test_optimize_criteo(tmpdir):
    input_path = os.path.join(DATA_START, "criteo/crit_orig")
    output_path = os.path.join(DATA_START, "criteo/crit_test_opt")

    notebook_path = os.path.join(dirname(TEST_PATH), "examples", "optimize_criteo.ipynb")
    _run_notebook(
        tmpdir, notebook_path, input_path, output_path, gpu_id=2,
    )


def test_rossman_example(tmpdir):
    pytest.importorskip("tensorflow")
    data_path = os.path.join(DATA_START, "rossman/data")
    input_path = os.path.join(DATA_START, "rossman/input")
    output_path = os.path.join(DATA_START, "rossman/output")

    notebookpre_path = os.path.join(
        dirname(TEST_PATH), "examples", "rossmann-store-sales-preproc.ipynb"
    )

    _run_notebook(
        tmpdir, notebookpre_path, data_path, input_path, gpu_id=1, clean_up=False,
    )

    notebookex_path = os.path.join(
        dirname(TEST_PATH), "examples", "rossmann-store-sales-example.ipynb"
    )
    _run_notebook(
        tmpdir, notebookex_path, input_path, output_path, gpu_id=1,
    )


def test_gpu_benchmark(tmpdir):
    input_path = os.path.join(DATA_START, "outbrains/input")
    output_path = os.path.join(DATA_START, "outbrains/output")

    notebook_path = os.path.join(dirname(TEST_PATH), "examples", "gpu_benchmark.ipynb")
    _run_notebook(tmpdir, notebook_path, input_path, output_path, gpu_id=0, batch_size=100000)


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
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if not os.path.exists(input_path):
        os.makedirs(input_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if batch_size:
        os.environ["BATCH_SIZE"] = str(batch_size)

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
    subprocess.check_output([sys.executable, script_path])

    # clear out products
    if clean_up:
        shutil.rmtree(output_path)
