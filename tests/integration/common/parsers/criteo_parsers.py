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

from .benchmark_parsers import BenchFastAI, Benchmark, create_bench_result


class CriteoBenchFastAI(BenchFastAI):
    def __init__(self, val=6, split=None):
        super().__init__("Criteo", val=val, split=split)

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
        super().__init__("HugeCTR")

    def get_epochs(self, output):
        epochs = []
        for line in output:
            if "AUC" in line:
                epochs.append(self.get_epoch(line))
        return epochs

    def get_epoch(self, line):
        split_line = line.split(",")
        iteration = int(split_line[0].split(":")[-1])
        auc = float(split_line[-1][:-2])
        bres_auc = create_bench_result(
            f"{self.name}_auc", [("iteration", iteration)], auc, "percent"
        )
        return [bres_auc]
