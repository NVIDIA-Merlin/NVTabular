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
import queue
import threading

import cudf
import cupy as cp

from nvtabular.workflow import BaseWorkflow
from nvtabular.io import _shuffle_gdf


def _num_steps(num_samples, step_size):
    return (num_samples - 1) // step_size + 1


class ChunkQueue:
    """
        This class takes partitions (parts) from an NVTabular dataset
        and concatenates them into a cudf dataframe "chunk". This chunk
        is subsequently transformed into its tensor representation using
        the iterator's transform.
        Parameters:
        num_parts: int, number of partitions from the iterator, an NVTabular Dataset,
                   to concatenate into a "chunk"
        batch_size: int, the number of records in each batch
        iterator: TensorBatchDatasetItr, the iterator to pull the partitions (parts) from
        shuffle: bool, enable/disable shuffling
        cats: [str], the list of categorical columns in the dataset
        conts: [str], the list of continuous columns in the dataset
        labels: [str], the list of label columns in the dataset
    """

    def __init__(
        self, batch_size, num_parts=1, shuffle=False,
    ):
        self.batch_size = batch_size
        self.num_parts = num_parts
        self.shuffle = shuffle
        self.q_out = queue.Queue(1)
        self._stop_event = threading.Event()

    @property
    def stopped(self):
        return self._stop_event.is_set()

    @property
    def empty(self):
        return self.q_out.empty()

    def get(self):
        return self.q_out.get()

    def batch(self, itr):
        current = []
        for value in itr:
            current.append(value)
            if len(current) == self.num_parts:
                yield current
                current = []
        if len(current) > 0:
            yield current

    def load_chunks(self, dev, itr, workflows, device_ctx):
        with device_ctx:
            spill = None
            for chunks in self.batch(itr):
                if self.stopped:
                    return

                if spill and not spill.empty:
                    chunks.insert(0, spill)

                chunks = cudf.core.reshape.concat(chunks)
                chunks.reset_index(drop=True, inplace=True)
                chunks, spill = self.get_batch_div_chunk(chunks)
                if self.shuffle:
                    _shuffle_gdf(chunks)

                num_samples = len(chunks)
                if num_samples > 0:
                    for workflow in workflows:
                        chunks = workflow.apply_ops(chunks)
                    self.q_out.put((chunks, num_samples))
                chunks = None

            # takes care final batch, which is less than batch size
            if spill:
                num_samples = len(spill)
                for workflow in workflows:
                    spill = workflow.apply_ops(spill)
                self.q_out.put((spill, num_samples))

    # For when an iterator is stopped before iteration is complete.
    def stop(self):
        self._stop_event.set()
        # TODO: should we be clearing? I can imagine a world where
        # you want the thread to stop but still want to grab
        # data out of the buffer
        self.q_out.queue.clear()

    def start(self):
        self._stop_event.clear()

    def get_batch_div_chunk(self, chunks):
        spill_idx = int(chunks.shape[0] / self.batch_size) * self.batch_size
        spill = cudf.DataFrame(chunks.iloc[spill_idx:])
        chunks = cudf.DataFrame(chunks.iloc[:spill_idx])
        if not chunks.empty:
            chunks.reset_index(drop=True, inplace=True)
        if not spill.empty:
            spill.reset_index(drop=True, inplace=True)
        return chunks, spill


def _validate_workflows(workflows, cat_names, cont_names, label_names):
    assert all([isinstance(w, BaseWorkflow) for w in workflows])
    # TODO: commenting out until it's clearer what the
    # columns in workflow.columns_cts["final"]["ctx"] mean
    # for workflow in workflows:
    #     assert set(workflow.columns_ctx["categorical"]["base"]) == set(cat_names)
    #     assert set(workflow.columns_ctx["continuous"]["base"]) == set(cont_names)
    #     assert set(workflow.columns_ctx["label"]["base"]) == set(label_names)

    #     cat_names = workflow.columns_ctx["final"]["ctx"]["categorical"]
    #     cont_names = workflow.columns_ctx["final"]["ctx"]["continuous"]
    #     label_name = workflow.columns_ctx["final"]["ctx"]["label"][0]
    return workflows


# TODO: implement as metaclass and assign methods to children
# to avoid having to do Dataset.<method> calls?
class DataLoader:
    def __init__(
        self,
        dataset,
        cat_names,
        cont_names,
        label_names,
        batch_size,
        shuffle,
        parts_per_chunk=1,
        workflows=None,
        devices=None,
    ):
        self.data = dataset
        self.indices = cp.arange(dataset.to_ddf().npartitions)

        devices = devices or [0]
        workflows = workflows or []
        self.workflows = _validate_workflows(workflows, cat_names, cont_names, label_names)

        self.cat_names = cat_names
        self.cont_names = cont_names
        self.label_names = label_names
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.devices = devices

        self._buff = ChunkQueue(batch_size=batch_size, num_parts=parts_per_chunk, shuffle=shuffle)
        self.chunk = None
        self._workers = None
        self._batch_idx = None
        self._num_steps = None

    def __len__(self):
        return _num_steps(self.data.num_rows, self.batch_size)

    @property
    def _working(self):
        if self._workers is not None:
            return any([t.is_alive() for t in self._workers])
        return False

    def __iter__(self):
        # we have some remaining workers from last iteration
        if self._workers is not None and self._working:
            # stop the buffer and wait for the workers to exit
            if not self._buff.stopped:
                self._buff.stop()
            for t in self._workers:
                t.join()

            # clear the remaining buffer
            self._buff.q_out.clear()

        # something happened that stopped our buffer,
        # reopen it
        if self._buff.stopped:
            self._buff.start()

        # shuffle partition indices to bring disparate
        # parts of the dataset "close" to one another
        if self.shuffle:
            cp.random.shuffle(self.indices)

        # build and start new threads for loading and
        # concatenating data
        self._workers = []
        for dev in self.devices:
            indices = self._gather_indices_for_dev(dev)
            itr = self.data.to_iter(indices=indices)

            t = threading.Thread(
                target=self._buff.load_chunks,
                args=(dev, itr, self.workflows, self._get_device_ctx(dev)),
            )
            t.daemon = True
            t.start()
            self._workers.append(t)
        return self

    def _get_device_ctx(self, dev):
        raise NotImplementedError

    def _create_tensors(self, gdf):
        raise NotImplementedError

    def _gather_indices_for_dev(self, dev):
        per_worker = int(len(self.indices) // len(self.devices)) + 1
        worker_id = self.devices.index(dev)
        start = worker_id * per_worker
        return self.indices[start : start + per_worker]

    def _get_next_chunk(self):
        chunk, num_samples = self._buff.get()
        self.chunk = self._create_tensors(chunk)
        self._num_steps = _num_steps(num_samples, self.batch_size)
        self._batch_idx = 0

    def __next__(self):
        # we've never initialized, do that now
        if self._workers is None:
            DataLoader.__iter__(self)

        # the buffer is empty and there are no running
        # threads to refill it: we must be empty
        if not self._working and self._buff.empty:
            self._workers = None
            self.chunk = None
            raise StopIteration

        # get a new chunk from the buffer
        if self.chunk is None:
            chunk = self._get_next_chunk()

        # slice the appropriate rows from each tensor
        slc = slice(self._batch_idx * self.batch_size, (self._batch_idx + 1) * self.batch_size)
        outputs = []
        for tensor in self.chunk:
            if isinstance(tensor, dict):
                outputs.append({name: x[slc] for name, x in tensor.items()})
            elif isinstance(tensor, list):
                outputs.append([x[slc] for x in tensor])
            elif tensor is not None:
                outputs.append(tensor[slc])
            else:
                outputs.append(tensor)

        # increment the batch index and get rid of our
        # self.chunk for memory purposes
        self._batch_idx += 1
        if self._batch_idx == self._num_steps:
            self.chunk = None
        return tuple(outputs)

    def map(self, workflow):
        # TODO: this is a bit ugly, how can we clean it up?
        # maybe think about doing some class consolidation
        workflows = self.workflows + [workflow]
        self.workflows = _validate_workflows(
            workflows, self.cat_names, self.cont_names, self.label_names
        )
        # TODO: update cat/cont/label names after
