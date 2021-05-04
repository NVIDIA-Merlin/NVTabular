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
import os
from uuid import uuid4

import numpy as np

from .writer import ThreadedWriter


class HugeCTRWriter(ThreadedWriter):
    def __init__(self, out_dir, suffix=".data", **kwargs):
        super().__init__(out_dir, **kwargs)
        self.suffix = suffix
        if self.use_guid:
            self.data_paths = [
                os.path.join(out_dir, f"{i}i.{uuid4().hex}{self.suffix}")
                for i in range(self.num_out_files)
            ]
        else:
            self.data_paths = [
                os.path.join(out_dir, f"{i}{self.suffix}") for i in range(self.num_out_files)
            ]
        self.data_writers = [open(f, "wb") for f in self.data_paths]
        # Reserve 64 bytes for header
        header = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.longlong)
        for i, writer in enumerate(self.data_writers):
            writer.write(header.tobytes())

    def _write_table(self, idx, data):
        # Prepare data format
        np_label = data[self.labels].to_pandas().astype(np.single).to_numpy()
        np_conts = data[self.conts].to_pandas().astype(np.single).to_numpy()
        nnz = np.intc(1)
        np_cats = data[self.cats].to_pandas().astype(np.uintc).to_numpy()
        # Write all the data samples
        for i, _ in enumerate(np_label):
            # Write Label
            self.data_writers[idx].write(np_label[i].tobytes())
            # Write conts (HugeCTR: dense)
            self.data_writers[idx].write(np_conts[i].tobytes())
            # Write cats (HugeCTR: Slots)
            for j, _ in enumerate(np_cats[i]):
                self.data_writers[idx].write(nnz.tobytes())
                self.data_writers[idx].write(np_cats[i][j].tobytes())

    def _write_thread(self):
        while True:
            item = self.queue.get()
            try:
                if item is self._eod:
                    break
                idx, data = item
                with self.write_locks[idx]:
                    self._write_table(idx, data)
            finally:
                self.queue.task_done()

    def _close_writers(self):
        for i, writer in enumerate(self.data_writers):
            if self.cats:
                # Write HugeCTR Metadata
                writer.seek(0)
                # error_check (0: no error check; 1: check_num)
                # num of samples in this file
                # Dimension of the labels
                # Dimension of the features
                # slot_num for each embedding
                # reserved for future use
                header = np.array(
                    [
                        0,
                        self.num_samples[i],
                        len(self.labels),
                        len(self.conts),
                        len(self.cats),
                        0,
                        0,
                        0,
                    ],
                    dtype=np.longlong,
                )
                writer.write(header.tobytes())
            writer.close()
        return None

    def _bytesio_to_disk(self):
        raise ValueError("hugectr binary format doesn't support PER_WORKER shuffle yet")
