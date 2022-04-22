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
    def get_info(self, output):
        bench_infos = []
        losses = []
        for line in output:
            if "run_time" in line:
                bench_infos.append(line)
            if "loss" in line:
                losses.append(line)
        loss_dict = {}
        if losses:
            loss_dict = {"loss": self.get_loss(losses[-1])}
        if bench_infos:
            bench_infos = self.get_dl_timing(bench_infos[-1:], optionals=loss_dict)
        return bench_infos

    def get_dl_thru(
        self, full_time, num_rows, epochs, throughput, optionals=None
    ) -> BenchmarkResult:
        metrics = [("thru", throughput), ("rows", num_rows), ("epochs", epochs)]
        optionals = optionals or {}
        for metric_name, metric_value in optionals.items():
            metrics.append((metric_name, metric_value))
        return create_bench_result(
            f"{self.name}_dataloader",
            metrics,
            full_time,
            "seconds",
        )

    def get_loss(self, line):
        return float(line)

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
        if time_format:
            x = time.strptime(r_time.split(",")[0], time_format)
            r_time = datetime.timedelta(
                hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec
            ).total_seconds()
        return create_bench_result(f"{self.name}_time", [("epoch", epoch)], r_time, "seconds")

    def aps(self, epoch, aps) -> BenchmarkResult:
        return create_bench_result(f"{self.name}_Avg_Prec", [("epoch", epoch)], aps, "percent")

    def get_dl_timing(self, output, optionals=None):
        timing_res = []
        for line in output:
            if line.startswith("run_time"):
                run_time, num_rows, epochs, dl_thru = line.split(" - ")
                run_time = float(run_time.split(": ")[1])
                num_rows = int(num_rows.split(": ")[1])
                epochs = int(epochs.split(": ")[1])
                dl_thru = float(dl_thru.split(": ")[1])
                bres = self.get_dl_thru(
                    run_time, num_rows * epochs, epochs, dl_thru, optionals=optionals
                )
                timing_res.append(bres)
        return timing_res[-1:]


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
            if "run_time" in line:
                epochs.append(self.get_dl_timing(line))
        return epochs[-1:]


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
    # only one entry because entries are split by Bench info
    new_results_list = results_list
    info_list = list(db.getInfo())
    if len(info_list) > 0:
        br_list = db.getResults(filterInfoObjList=[bench_info])
        if br_list:
            br_list = br_list[0][1]
        results_to_remove = []
        for result in results_list:
            if any(br.funcName == result.funcName for br in br_list):
                results_to_remove.append(result)
        new_results_list = [result for result in results_list if result not in results_to_remove]
        # breakpoint()
    for results in new_results_list:
        if isinstance(results, list):
            for result in results:
                db.addResult(bench_info, result)
        else:
            db.addResult(bench_info, results)


def create_bench_result(name, arg_tuple_list, result, unit):
    return BenchmarkResult(
        funcName=name, argNameValuePairs=arg_tuple_list, unit=unit, result=result
    )
