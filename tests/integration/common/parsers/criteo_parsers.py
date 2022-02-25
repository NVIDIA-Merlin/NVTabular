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
import re

from tests.integration.common.parsers.benchmark_parsers import (
    BenchFastAI,
    Benchmark,
    StandardBenchmark,
    create_bench_result,
)

decimal_regex = "[0-9]+\.?[0-9]*|\.[0-9]+"  # noqa


class CriteoBenchFastAI(BenchFastAI):
    def __init__(self, val=6, split=None):
        super().__init__("CriteoFastAI", val=val, split=split)

    def get_epoch(self, line):
        epoch, t_loss, v_loss, roc, aps, o_time = line.split()
        t_loss = self.loss(epoch, float(t_loss))
        v_loss = self.loss(epoch, float(v_loss), l_type="valid")
        roc = self.roc_auc(epoch, float(roc))
        aps = self.aps(epoch, float(aps))
        o_time = self.time(epoch, o_time, time_format="%H:%M:%S")
        return [t_loss, v_loss, roc, aps, o_time]


class CriteoBenchHugeCTR(Benchmark):
    def __init__(self):
        super().__init__("CriteoHugeCTR")

    def get_epochs(self, output):
        epochs = []
        for line in output:
            if "AUC" in line:
                epochs.append(self.get_epoch(line))
        return epochs[-1:]

    def get_epoch(self, line):
        split_line = line.split(",")
        iteration = int(split_line[0].split(":")[-1])
        auc = float(split_line[-1][:-2])
        bres_auc = create_bench_result(
            f"{self.name}_auc", [("iteration", iteration)], auc, "percent"
        )
        return [bres_auc]


class CriteoTensorflow(StandardBenchmark):
    def __init__(self, name="CriteoTensorflow"):
        super().__init__(name)

    def get_info(self, output):
        bench_infos = []
        for line in output:
            if "run_time" in line:
                bench_infos.extend(self.get_timing_line(line))
            if "loss" in line:
                bench_infos.extend(self.get_loss(line))
        return bench_infos[-1:]

    def get_timing_line(self, line):
        runtime, epochs, rows, dl_thru = line.split("-")
        epochs = int(epochs.split(":")[-1])
        runtime = self.time(epochs, float(runtime.split(":")[-1]), time_format=None)

        dl_thru = float(dl_thru.split(":")[-1])
        bres_dl_thru = self.get_dl_thru(runtime, rows, epochs, dl_thru)
        return [bres_dl_thru, runtime]

    def get_loss(self, line):
        loss = line.split("-")[-1]
        loss = loss.split(":")[-1]
        losses = re.findall(decimal_regex, loss)
        if losses:
            loss = float(losses[-1])
            br_loss = self.loss(1, loss)
            return [br_loss]
        return []


class CriteoTorch(StandardBenchmark):
    def __init__(self, name="CriteoTorch"):
        super().__init__(name)

    def get_info(self, output):
        bench_infos = []
        for line in output:
            if "run_time" in line:
                bench_infos.extend(self.get_timing_line(line))
            if "loss" in line and "Train" in line and "Valid" in line:
                bench_infos.extend(self.get_loss(line))
        return bench_infos[-1:]

    def get_timing_line(self, line):
        # run_time: 12.725089311599731 - rows: 2292 - epochs: 0 - dl_thru: 180.1166140272741
        runtime, rows, epochs, dl_thru = line.split("-")
        epochs = int(epochs.split(":")[-1])
        runtime = self.time(epochs, float(runtime.split(":")[-1]), time_format=None)
        dl_thru = float(dl_thru.split(":")[-1])
        bres_dl_thru = self.get_dl_thru(runtime, rows, epochs, dl_thru)
        return [bres_dl_thru, runtime]

    def get_loss(self, line):
        # Epoch 00. Train loss: 0.1944. Valid loss: 0.1696.
        loss_parse = line.split(". ")
        epoch = loss_parse[0].split(" ")[-1]
        train_loss = loss_parse[1].split(":")[-1]
        valid_loss = loss_parse[2].split(":")[-1]

        epoch = re.findall(decimal_regex, epoch)[-1]
        train_loss = re.findall(decimal_regex, train_loss)[-1]
        valid_loss = re.findall(decimal_regex, valid_loss)[-1]

        epoch = int(epoch)
        train_loss = float(train_loss)
        valid_loss = float(valid_loss)
        t_br_loss = self.loss(epoch, train_loss, l_type="train")
        v_br_loss = self.loss(epoch, valid_loss, l_type="valid")

        return [t_br_loss, v_br_loss]
