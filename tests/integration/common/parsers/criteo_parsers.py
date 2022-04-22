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
    StandardBenchmark,
    create_bench_result,
)

decimal_regex = "[0-9]+\.?[0-9]*|\.[0-9]+"  # noqa pylint: disable=W1401


class CriteoBenchFastAI(BenchFastAI):
    def __init__(self, name="CriteoFastAI", val=6, split=None):
        self.name = name
        self.val = val
        self.split = split

    def get_info(self, output):
        bench_infos = []
        losses = []
        for line in output:
            if "run_time" in line:
                bench_infos.append(line)
            if "loss" in line and "Train" in line and "Valid" in line:
                losses.append(line)
        loss_dict = {}
        if losses:
            for loss in losses:
                t_loss, v_loss = self.get_loss(loss)
                loss_dict["loss_train"] = t_loss
                loss_dict["loss_valid"] = v_loss
        if bench_infos:
            bench_infos = self.get_dl_timing(bench_infos[-1:], optionals=loss_dict)
        return bench_infos

    def get_epoch(self, line):
        epoch, t_loss, v_loss, roc, aps, o_time = line.split()
        t_loss = self.loss(epoch, float(t_loss))
        v_loss = self.loss(epoch, float(v_loss), l_type="valid")
        roc = self.roc_auc(epoch, float(roc))
        aps = self.aps(epoch, float(aps))
        return [t_loss, v_loss, roc, aps, o_time]

    def get_loss(self, line):
        epoch, t_loss, v_loss, roc, aps, o_time = line.split()
        t_loss = float(t_loss)
        v_loss = float(v_loss)
        return [t_loss, v_loss]


class CriteoBenchHugeCTR(StandardBenchmark):
    def __init__(self, name="CriteoHugeCTR"):
        self.name = name

    def get_epochs(self, output):
        aucs = []
        for line in output:
            if "AUC:" in line:
                auc_num = float(line.split("AUC:")[-1])
                aucs.append(auc_num)
            if "run_time:" in line:
                run_time = self.get_runtime(line)
        if run_time and aucs:
            return self.get_epoch(max(aucs), run_time)
        return []

    def get_runtime(self, line):
        split_line = line.split(":")
        return float(split_line[1])

    def get_epoch(self, auc, runtime):
        bres_auc = create_bench_result(f"{self.name}_auc", [("time", runtime)], auc, "percent")
        return [bres_auc]


class CriteoTensorflow(StandardBenchmark):
    def __init__(self, name="CriteoTensorFlow"):
        self.name = name

    def get_loss(self, line):
        loss = line.split("-")[-1]
        loss = loss.split(":")[-1]
        losses = re.findall(decimal_regex, loss)
        losses = losses or []
        return float(losses[-1])


class CriteoTorch(StandardBenchmark):
    def __init__(self, name="CriteoTorch"):
        self.name = name

    def get_info(self, output):
        bench_infos = []
        losses = []
        for line in output:
            if "run_time" in line:
                bench_infos.append(line)
            if "loss" in line and "Train" in line and "Valid" in line:
                losses.append(line)
        loss_dict = {}
        if losses:
            for idx, loss in enumerate(losses):
                t_loss, v_loss = self.get_loss(loss)
                loss_dict["loss_train"] = t_loss
                loss_dict["loss_valid"] = v_loss
        if bench_infos:
            bench_infos = self.get_dl_timing(bench_infos[-1:], optionals=loss_dict)
        return bench_infos

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
        return [train_loss, valid_loss]
