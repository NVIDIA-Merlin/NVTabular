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
DATA_START = os.environ.get("DATASET_DIR", "/raid/criteo")


def test_criteo_notebook(db, bench_info, tmpdir):
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
    send_results(db, bench_info, bench_results)
    


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
    bench_results = RossBenchFastAI().get_epochs(out.splitlines())
    send_results(db, bench_info, bench_results)


    notebookex_path = os.path.join(
        dirname(TEST_PATH), "examples/rossmann", "rossmann-store-sales-pytorch.ipynb"
    )
    out = _run_notebook(tmpdir, notebookex_path, input_path, output_path, gpu_id=1)
    bench_results = RossBenchPytorch().get_epochs(out.splitlines())
    send_results(db, bench_info, bench_results)

    
    notebookex_path = os.path.join(
        dirname(TEST_PATH), "examples/rossmann", "rossmann-store-sales-tensorflow.ipynb"
    )
    out = _run_notebook(tmpdir, notebookex_path, input_path, output_path, gpu_id=1)
    bench_results = RossBenchTensorFlow().get_epochs(out.splitlines())
    send_results(db, bench_info, bench_results)


# def test_gpu_benchmark(tmpdir):
#     input_path = os.path.join(DATA_START, "outbrains/input")
#     output_path = os.path.join(DATA_START, "outbrains/output")

#     notebook_path = os.path.join(dirname(TEST_PATH), "examples", "gpu_benchmark.ipynb")
#     _run_notebook(tmpdir, notebook_path, input_path, output_path, gpu_id=0, batch_size=100000)


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
        with open(f"test_res_{note_name}", "w+") as w_file:
            w_file.write(output)
    # clear out products
    if clean_up:
        shutil.rmtree(output_path)
    return output



    



class Benchmark():
    def __init__(self, target_id, val=1, split=None):
        self.name = f"{target_id}"
        self.val = val
        self.split = split
    
    def bres_loss(self, epoch, loss, l_type="train"):
        return create_bench_result(f"{self.name}_{l_type}_loss",
                                   [("epoch", epoch)],
                                   loss,
                                   "percent")


    def bres_rmspe(self, epoch, rmspe):
        return create_bench_result(f"{self.name}_exp_rmspe",
                                   [("epoch", epoch)],
                                   rmspe,
                                   "percent")
    
    def bres_acc(self, epoch, acc):
        return create_bench_result(f"{self.name}_exp_rmspe",
                                   [("epoch", epoch)],
                                   acc,
                                   "percent")

    def bres_roc_auc(self, epoch, acc):
        return create_bench_result(f"{self.name}_exp_rmspe",
                                   [("epoch", epoch)],
                                   acc,
                                   "percent")
    
    
    def bres_time(self, epoch, r_time, time_format='%M:%S'):
        x = time.strptime(r_time.split(',')[0],time_format)
        r_time = datetime.timedelta(
                    hours=x.tm_hour,
                    minutes=x.tm_min,
                    seconds=x.tm_sec
                ).total_seconds()
        return create_bench_result(f"{self.name}_time",
                                   [("epoch", epoch)],
                                   r_time,
                                   "seconds")
    def bres_aps(self, epoch, aps):
        return create_bench_result(f"{self.name}_Avg_Prec",
                                   [("epoch", epoch)],
                                   aps,
                                   "percent")

    def get_epoch(self, line):
        raise NotImplementedError("Must Define logic for parsing metrics per epoch")
        
    def get_epochs(self, output):
        raise NotImplementedError("Must Define logic for parsing output")

        
class RossBenchTensorFlow(Benchmark):
    def __init__(self, split=" - "):
        super().__init__(f"Rossmann_tf", split=split)
        
    def get_epoch(self, line, epoch=0):
        _, _, t_loss, t_rmspe = line.split(self.split)
        t_loss = self.bres_loss(epoch, float(t_loss.split(": ")[1]))
#         v_loss = self.bres_loss(epoch, float(v_loss.split(": ")[1]), l_type="valid")
        t_rmspe = self.bres_rmspe(epoch, float(t_rmspe.split(": ")[1]))
#         v_rmspe = self.bres_rmspe(epoch, float(v_rmspe.split(": ")[1]))
#         return [t_loss, v_loss, t_rmspe, v_rmspe]
        return [t_loss, t_rmspe]
            
        
    def get_epochs(self, output):
        epochs = []
        for idx, line in enumerate(output):
            if "Epoch" in line:
                epoch = int(line.split()[-1].split("/")[0])
                # output skips line for formatting and remove returns (\x08)
                content_line = output[idx + 2].rstrip("\x08")
                # epoch line, detected based on if 1st character is a number
                post_evts = self.get_epoch(content_line, epoch=epoch)
                epochs.append(post_evts)
        return epochs


class RossBenchPytorch(Benchmark):
    def __init__(self, split=". "):
        super().__init__(f"Rossmann_torch", split=split)
        
        
    def get_epoch(self, line):
        epoch, t_loss, t_rmspe, v_loss, v_rmspe = line.split(self.split)
        epoch = epoch.split()[1]
        t_loss = self.bres_loss(epoch, float(t_loss.split(": ")[1]))
        v_loss = self.bres_loss(epoch, float(v_loss.split(": ")[1]), l_type="valid")
        t_rmspe = self.bres_rmspe(epoch, float(t_rmspe.split(": ")[1]))
        v_rmspe = self.bres_rmspe(epoch, float(v_rmspe.split(": ")[1].split(".")[0]))
        return [t_loss, v_loss, t_rmspe, v_rmspe]
            
        
    def get_epochs(self, output):
        epochs = []
        for line in output:
            if "Epoch" in line:
                # epoch line, detected based on if 1st character is a number
                post_evts = self.get_epoch(line)
                epochs.append(post_evts)
        return epochs

        
class BenchFastAI(Benchmark):
    def __init__(self, target_id, val=6, split=None):
        super().__init__(f"{target_id}_fastai", val=val, split=split)
    
    def get_epochs(self, output):
        epochs = []
        for line in output:
            split_line = line.split(self.split) if self.split else line.split()
            if len(split_line) == self.val and is_number(split_line[0]):
                # epoch line, detected based on if 1st character is a number
                post_evts = self.get_epoch(line)
                epochs.append(post_evts)
        return epochs




class CriteoBenchFastAI(BenchFastAI):
    def __init__(self, val=6, split=None):
        super().__init__("Criteo", val=val, split=split)
        
    def get_epoch(self, line):
        epoch, t_loss, v_loss, roc, aps, o_time = line.split()
        t_loss = self.bres_loss(epoch, float(t_loss))
        v_loss = self.bres_loss(epoch, float(v_loss), l_type="valid")
        roc = self.bres_roc_auc(epoch, float(roc))
        aps = self.bres_aps(epoch, float(aps))
        o_time = self.bres_time(epoch, o_time)
        return [t_loss, v_loss, roc, aps, o_time]


class RossBenchFastAI(BenchFastAI):
    def __init__(self, val=5, split=None):
        super().__init__("Rossmann", val=val, split=split)
        
    def get_epoch(self, line):
        epoch, t_loss, v_loss, exp_rmspe, o_time = line.split()
        t_loss = self.bres_loss(epoch, float(t_loss))
        v_loss = self.bres_loss(epoch, float(v_loss), l_type="valid")
        exp_rmspe = self.bres_rmspe(epoch, float(exp_rmspe))
        o_time = self.bres_time(epoch, o_time)
        return [t_loss, v_loss, exp_rmspe, o_time]
    
        


def is_number(str_to_num):
    try:
        val = int(str_to_num)
        return True
    except ValueError:
        return False
                
def send_results(db, bench_info, results_list):
    for results in results_list:
        if isinstance(results, list):
            for result in results:
                db.addResult(bench_info, result)
        else:
            db.addResult(bench_info, results)

def create_bench_result(name, arg_tuple_list, result, unit):
    return BenchmarkResult(funcName=name,
                           argNameValuePairs=arg_tuple_list,
                           unit=unit,
                           result=result)