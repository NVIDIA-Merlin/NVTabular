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
import os

import numpy as np

from .writer import ThreadedWriter


class HugeCTRWriter(ThreadedWriter):
    def __init__(self, out_dir, **kwargs):
        super().__init__(out_dir, **kwargs)
        self.data_paths = [os.path.join(out_dir, f"{i}.data") for i in range(self.num_out_files)]
        self.data_writers = [open(f, "wb") for f in self.data_paths]
        # Reserve 64 bytes for header
        header = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.longlong)
        for i, writer in enumerate(self.data_writers):
            writer.write(header.tobytes())

    def _write_table(self, idx, data):
        df = data[self.labels].to_pandas().astype(np.single)
        df[self.cats] = data[self.cats].to_pandas().astype(np.intc)
        df[self.conts] = data[self.conts].to_pandas().astype(np.single)
        self.data_writers[idx].write(df.to_records(index=False).tobytes())

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
