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

import datetime
import time

from asvdb import BenchmarkResult


class Benchmark:
    """
    Main general benchmark parsing class
    """

    def __init__(self, target_id, val=1, split=None):
        self.name = f"{target_id}"
        self.val = val
        self.split = split

    def get_epoch(self, line):
        raise NotImplementedError("Must Define logic for parsing metrics per epoch")

    def get_epochs(self, output):
        raise NotImplementedError("Must Define logic for parsing output")


# Sub classes


class StandardBenchmark(Benchmark):
    def get_dl_thru(self, full_time, num_rows, epochs, throughput) -> BenchmarkResult:
        return create_bench_result(
            f"{self.name}_dataloader",
            [("time", full_time), ("rows", num_rows), ("epochs", epochs)],
            throughput,
            "rows/second",
        )

    def loss(self, epoch, loss, l_type="train") -> BenchmarkResult:
        return create_bench_result(
            f"{self.name}_{l_type}_loss", [("epoch", epoch)], loss, "percent"
        )

    def rmspe(self, epoch, rmspe) -> BenchmarkResult:
        return create_bench_result(f"{self.name}_exp_rmspe", [("epoch", epoch)], rmspe, "percent")

    def acc(self, epoch, acc) -> BenchmarkResult:
        return create_bench_result(f"{self.name}_exp_rmspe", [("epoch", epoch)], acc, "percent")

    def roc_auc(self, epoch, acc) -> BenchmarkResult:
        return create_bench_result(f"{self.name}_exp_rmspe", [("epoch", epoch)], acc, "percent")

    def time(self, epoch, r_time, time_format="%M:%S") -> BenchmarkResult:
        x = time.strptime(r_time.split(",")[0], time_format)
        r_time = datetime.timedelta(
            hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec
        ).total_seconds()
        return create_bench_result(f"{self.name}_time", [("epoch", epoch)], r_time, "seconds")

    def aps(self, epoch, aps) -> BenchmarkResult:
        return create_bench_result(f"{self.name}_Avg_Prec", [("epoch", epoch)], aps, "percent")

    def get_dl_timing(self, output):
        timing_res = []
        for line in output:
            if line.startswith("run_time"):
                run_time, num_rows, epochs, dl_thru = line.split(" - ")
                run_time = float(run_time.split(": ")[1])
                num_rows = int(num_rows.split(": ")[1])
                epochs = int(epochs.split(": ")[1])
                dl_thru = float(dl_thru.split(": ")[1])
                bres = self.get_dl_thru(run_time, num_rows, epochs, dl_thru)
                timing_res.append(bres)
        return timing_res


class BenchFastAI(StandardBenchmark):
    def __init__(self, target_id, val=6, split=None):
        super().__init__(f"{target_id}_fastai", val=val, split=split)

    def get_epochs(self, output):
        epochs = []
        for line in output:
            split_line = line.split(self.split) if self.split else line.split()
            if len(split_line) == self.val and is_whole_number(split_line[0]):
                # epoch line, detected based on if 1st character is a number
                post_evts = self.get_epoch(line)
                epochs.append(post_evts)
        return epochs


# Utils


def is_whole_number(str_to_num):
    try:
        int(str_to_num)
        return True
    except ValueError:
        return False


def is_float(str_to_flt):
    try:
        float(str_to_flt)
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
    return BenchmarkResult(
        funcName=name, argNameValuePairs=arg_tuple_list, unit=unit, result=result
    )
