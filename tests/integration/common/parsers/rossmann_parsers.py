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

from .benchmark_parsers import BenchFastAI, StandardBenchmark


class RossBenchTensorFlow(StandardBenchmark):
    def __init__(self, split=" - "):
        super().__init__("Rossmann_tf", split=split)

    def get_epoch(self, line, epoch=0):
        _, _, t_loss, t_rmspe = line.split(self.split)
        t_loss = self.loss(epoch, float(t_loss.split(": ")[1]))
        t_rmspe = self.rmspe(epoch, float(t_rmspe.split(": ")[1]))
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


class RossBenchPytorch(StandardBenchmark):
    def __init__(self, split=". "):
        super().__init__("Rossmann_torch", split=split)

    def get_epoch(self, line):
        epoch, t_loss, t_rmspe, v_loss, v_rmspe = line.split(self.split)
        epoch = epoch.split()[1]
        t_loss = self.loss(epoch, float(t_loss.split(": ")[1]))
        v_loss = self.loss(epoch, float(v_loss.split(": ")[1]), l_type="valid")
        t_rmspe = self.rmspe(epoch, float(t_rmspe.split(": ")[1]))
        v_rmspe = self.rmspe(epoch, float(v_rmspe.split(": ")[1].split(".")[0]))
        return [t_loss, v_loss, t_rmspe, v_rmspe]

    def get_epochs(self, output):
        epochs = []
        for line in output:
            if "Epoch" in line:
                # epoch line, detected based on if 1st character is a number
                post_evts = self.get_epoch(line)
                epochs.append(post_evts)
        return epochs


class RossBenchFastAI(BenchFastAI):
    def __init__(self, val=5, split=None):
        super().__init__("Rossmann", val=val, split=split)

    def get_epoch(self, line):
        epoch, t_loss, v_loss, exp_rmspe, o_time = line.split()
        t_loss = self.loss(epoch, float(t_loss))
        v_loss = self.loss(epoch, float(v_loss), l_type="valid")
        exp_rmspe = self.rmspe(epoch, float(exp_rmspe))
        o_time = self.time(epoch, o_time)
        return [t_loss, v_loss, exp_rmspe, o_time]
