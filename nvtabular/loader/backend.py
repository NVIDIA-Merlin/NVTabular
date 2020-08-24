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
from nvtabular.ops import _get_embedding_order


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
        self, qsize, num_parts=1, shuffle=False, put_wait=1e-6,
    ):
        self.num_parts = num_parts
        self.shuffle = shuffle
        self.put_wait = put_wait
        self.q_out = queue.Queue(qsize)
        self._stop_event = threading.Event()

    @property
    def stopped(self):
        return self._stop_event.is_set()

    @property
    def empty(self):
        return self.q_out.empty()

    def get(self):
        return self.q_out.get()

    def put(self, packet):
        while True:
            if self.stopped:
                return True

            try:
                self.q_out.put(packet, timeout=self.put_wait)
                return False
            except queue.Full:
                continue

    def batch(self, itr):
        """
        iterates through gpu_mem_frac size chunks of dataset
        and concatenates every `num_parts` of them.
        """
        current = []
        for value in itr:
            current.append(value)
            if len(current) == self.num_parts:
                yield current
                current = []
        if len(current) > 0:
            yield current

    def load_chunks(self, dev, dataloader):
        indices = dataloader._gather_indices_for_dev(dev)
        itr = dataloader.data.to_iter(indices=indices)

        with dataloader._get_device_ctx(dev):
            spill = None
            for chunks in self.batch(itr):
                if self.stopped:
                    return

                if spill and not spill.empty:
                    chunks.insert(0, spill)

                chunks = cudf.core.reshape.concat(chunks)
                chunks.reset_index(drop=True, inplace=True)
                chunks, spill = self.get_batch_div_chunk(chunks, dataloader.batch_size)
                if self.shuffle:
                    _shuffle_gdf(chunks)

                num_samples = len(chunks)
                if num_samples > 0:
                    for workflow in dataloader.workflows:
                        chunks = workflow.apply_ops(chunks)

                    # map from big chunk to fraemwork specific tensors
                    chunks = dataloader._create_tensors(chunks)

                    # split them into batches
                    chunks = [dataloader._create_batch(x, num_samples) for x in chunks]
                    chunks = zip(*chunks)

                    # put returns True if buffer is stopped before
                    # packet can be put in queue. Keeps us from
                    # freezing on a put on a full queue
                    if self.put(chunks):
                        return
                chunks = None

            # takes care final batch, which is less than batch size
            if spill:
                for workflow in dataloader.workflows:
                    spill = workflow.apply_ops(spill)
                spill = dataloader._create_tensors(chunks)
                self.put(spill)

    # For when an iterator is stopped before iteration is complete.
    def stop(self):
        self._stop_event.set()
        # TODO: should we be clearing? I can imagine a world where
        # you want the thread to stop but still want to grab
        # data out of the buffer
        self.q_out.queue.clear()

    def start(self):
        self._stop_event.clear()

    def get_batch_div_chunk(self, chunks, batch_size):
        # TODO: is there a way to do this using cupy?
        spill_idx = int(chunks.shape[0] / batch_size) * batch_size
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

        self._buff = ChunkQueue(len(devices), num_parts=parts_per_chunk, shuffle=shuffle)
        self._batch_itr = None
        self._workers = None

    def __len__(self):
        return _num_steps(self.data.num_rows, self.batch_size)

    @property
    def _working(self):
        if self._workers is not None:
            return any([t.is_alive() for t in self._workers])
        return False

    def stop(self):
        # TODO: raise warning or even error if condition
        # isn't met?
        if self._workers is not None and self._working:
            if not self._buff.stopped:
                self._buff.stop()
            for t in self._workers:
                t.join()
            self._buff.q_out.clear()

    def _gather_indices_for_dev(self, dev):
        per_worker = int(len(self.indices) // len(self.devices)) + 1
        worker_id = self.devices.index(dev)
        start = worker_id * per_worker
        return self.indices[start : start + per_worker]

    def __iter__(self):
        self.stop()
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
            t = threading.Thread(target=self._buff.load_chunks, args=(dev, self))
            t.daemon = True
            t.start()
            self._workers.append(t)
        return self

    def __next__(self):
        # we've never initialized, do that now
        # need this because tf.keras.Model.fit will
        # call next() cold
        if self._workers is None:
            DataLoader.__iter__(self)

        # get the first chunks
        if self._batch_itr is None:
            chunks = self._buff.get()
            self._batch_itr = iter(chunks)

        # try to iterate through existing batches
        try:
            cats, conts, labels = next(self._batch_itr)
        except StopIteration:
            # if current chunk is exhausted, check if we
            # anticipate any more chunks getting created
            # if not, raise the StopIteration
            if not self._working and self._buff.empty:
                self._workers = None
                self._batch_itr = None
                raise StopIteration

            # otherwise get the next chunks and return
            # the first batch
            chunks = self._buff.get()
            self._batch_itr = iter(chunks)
            cats, conts, labels = next(self._batch_itr)
        return self._handle_tensors(cats, conts, labels)

    def map(self, workflow):
        """
        Map an NVTabular Workflow on to a data loader to do
        online preprocessing
        """
        workflows = self.workflows + [workflow]
        self.workflows = _validate_workflows(
            workflows, self.cat_names, self.cont_names, self.label_names
        )
        # TODO: update cat/cont/label names after

    def _get_segment_lengths(self, num_samples):
        """
        Helper function to build indices to pass
        to <torch|tf>.split functions for breaking
        up into batches
        """
        idx = [self.batch_size for _ in range(num_samples // self.batch_size)]
        idx.append(num_samples % self.batch_size)
        return idx

    def _to_tensor(self, gdf, dtype=None):
        """
        One of the mandatory functions a child class needs
        to implement. Maps from a cudf DataFrame to a
        tensor in the appropriate library, with an optional
        dtype kwarg to do explicit casting if need be
        """
        raise NotImplementedError

    def _get_device_ctx(self, dev):
        """
        One of the mandatory functions a child class needs
        to implement. Maps from a GPU index to a framework
        context object for placing tensors on specific GPUs
        """
        raise NotImplementedError

    def _create_batch(self, tensor, num_samples):
        """
        One of the mandatory functions a child class needs
        to implement. Splits a `tensor` with `num_samples`
        rows into a list of tensors with `batch_size` rows
        """
        # TODO: can we just do this with some sort of
        # self._split_fn attribute?
        raise NotImplementedError

    def _create_tensors(self, gdf):
        """
        Breaks a dataframe down into the relevant
        categorical, continuous, and label tensors.
        Can be overrideen
        """
        # TODO: how will this work once we have multi-hots
        # also seems brittle to labels with mixed type
        gdf_cats, gdf_conts, gdf_label = (
            gdf[_get_embedding_order(self.cat_names)],
            gdf[self.cont_names],
            gdf[self.label_names],
        )
        del gdf
        cats = self._to_tensor(gdf_cats)
        conts = self._to_tensor(gdf_conts)
        label = self._to_tensor(gdf_label)

        del gdf_cats, gdf_conts, gdf_label
        return [cats, conts, label]

    def _handle_tensors(self, cats, conts, labels):
        return cats, conts, labels
