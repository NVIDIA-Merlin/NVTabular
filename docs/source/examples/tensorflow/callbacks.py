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

import time

import tensorflow as tf


class ThroughputLogger(tf.keras.callbacks.Callback):
    def __init__(self, batch_size, window_size=10, **kwargs):
        self.batch_size = batch_size
        self.window_size = window_size
        self.times = []
        super(ThroughputLogger, self).__init__(**kwargs)

    def on_epoch_begin(self, epoch, logs=None):
        self.times = [time.time()]

    def on_batch_end(self, batch, logs=None):
        self.times.append(time.time())
        if len(self.times) > self.window_size:
            del self.times[0]

        time_delta = self.times[-1] - self.times[0]
        tf.summary.scalar("throughput", self.batch_size * len(self.times) / time_delta, step=batch)
