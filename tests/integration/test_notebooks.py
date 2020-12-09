#
# Copyright (c) 2020, NVIDIA CORPORATION.
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
import ast
import re
import time
import datetime
from os.path import dirname, realpath

import pytest
from asvdb import BenchmarkResult

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
        gpu_id=0,
        batch_size=100000,
    )


def test_criteohugectr_notebook(tmpdir):
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


def test_optimize_criteo(tmpdir):
    input_path = os.path.join(DATA_START, "criteo/crit_orig")
    output_path = os.path.join(DATA_START, "criteo/crit_test_opt")

    notebook_path = os.path.join(dirname(TEST_PATH), "examples", "optimize_criteo.ipynb")
    _run_notebook(tmpdir, notebook_path, input_path, output_path, gpu_id=2)


def test_rossman_example(tmpdir, bench_info, db):
    pytest.importorskip("tensorflow")
    data_path = os.path.join(DATA_START, "rossman/data")
    input_path = os.path.join(DATA_START, "rossman/input")
    output_path = os.path.join(DATA_START, "rossman/output")

    notebookpre_path = os.path.join(
        dirname(TEST_PATH), "examples/rossmann", "rossmann-store-sales-preproc.ipynb"
    )

    out = _run_notebook(tmpdir, notebookpre_path, data_path, input_path, gpu_id=1, clean_up=False)

    notebookpre_path = os.path.join(
        dirname(TEST_PATH), "examples/rossmann", "rossmann-store-sales-feature-engineering.ipynb"
    )

    out = _run_notebook(tmpdir, notebookpre_path, data_path, input_path, gpu_id=1, clean_up=False)
    
    notebookex_path = os.path.join(
        dirname(TEST_PATH), "examples/rossmann", "rossmann-store-sales-fastai.ipynb"
    )
    out = _run_notebook(tmpdir, notebookex_path, input_path, output_path, gpu_id=1)
    bench_results = bench_fastai("rossmann").get_epochs(out.splitlines())
    for results in bench_results:
        for result in results:
            db.addResult(bench_info, result)


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
    output = subprocess.check_output([sys.executable, script_path])
    # save location will default to run location
    output = output.decode("utf-8") 
    _,  note_name = os.path.split(notebook_path)
    note_name = note_name.split(".")[0]
    if output:
#         re_t = re.compile(r"^[^<>/{}[\]~`]*$")
        with open(f"test_res_{note_name}", "w+") as w_file:
            w_file.write(output)
    # clear out products
    if clean_up:
        shutil.rmtree(output_path)
    return output

        
class bench_fastai():
    def __init__(self, target_id):
        self.name = f"{target_id}_train_fastai"
    
    def bres_t_loss(self, epoch, t_loss):
        b_res = BenchmarkResult(funcName=f"{self.name}_train_loss",
                           argNameValuePairs=[
                              ("epoch", epoch)
                           ],
                           unit='percent',
                           result=t_loss)
        return b_res

    def bres_v_loss(self, epoch, v_loss):
        b_res = BenchmarkResult(funcName=f"{self.name}_valid_loss",
                           argNameValuePairs=[
                              ("epoch", epoch)
                           ],
                           unit='percent',
                           result=v_loss)
        return b_res

    def bres_rmspe(self, epoch, rmspe):
        b_res = BenchmarkResult(funcName=f"{self.name}_exp_rmspe",
                           argNameValuePairs=[
                              ("epoch", epoch)
                           ],
                           unit='percent',
                           result=rmspe)
        return b_res

    def bres_time(self, epoch, r_time):
        x = time.strptime(r_time.split(',')[0],'%M:%S')
        r_time = datetime.timedelta(
                    hours=x.tm_hour,
                    minutes=x.tm_min,
                    seconds=x.tm_sec
                ).total_seconds()
        b_res = BenchmarkResult(funcName=f"{self.name}_time",
                           argNameValuePairs=[
                              ("epoch", epoch)
                           ],
                           unit='seconds',
                           result=r_time)
        return b_res        
    
    
    def get_epoch(self, line):
        epoch, t_loss, v_loss, exp_rmspe, o_time = line.split()
        t_loss = self.bres_t_loss(epoch, float(t_loss))
        v_loss = self.bres_v_loss(epoch, float(v_loss))
        exp_rmspe = self.bres_rmspe(epoch, float(exp_rmspe))
        o_time = self.bres_time(epoch, o_time)
        return [t_loss, v_loss, exp_rmspe, o_time]
    
    
    def get_epochs(self, output):
        epochs = []
        for line in output:
            split_line = line.split()
            if len(split_line) > 1 and is_number(split_line[0]):
                # epoch line, detected based on if 1st character is a number
                post_evts = self.get_epoch(line)
                epochs.append(post_evts)
        return epochs

def is_number(str_to_num):
    try:
        val = int(str_to_num)
        return True
    except ValueError:
        return False
                

    