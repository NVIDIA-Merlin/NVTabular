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
        self,
        num_parts=1,
        batch_size=None,
        shuffle=False,
        # TODO: these should be moved to attributes
        # of the `itr` (which ideally could even just
        # be replaced by methods from the DataLoader)
        cat_cols=None,
        cont_cols=None,
        label_cols=None,
    ):
        self.num_parts = num_parts
        self.batch_size = batch_size
        self.q_out = queue.Queue(1)
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.label_cols = label_cols
        self.shuffle = shuffle
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

    def load_chunks(self, dev, itr):
        with itr.device_ctx(dev):
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
                    chunks = itr.preprocess(chunks)
                    chunks = itr.create_tensors(
                        chunks,
                        cat_names=self.cat_cols,
                        cont_names=self.cont_cols,
                        label_names=self.label_cols,
                    )
                    # chunks tensorized
                    self.q_out.put((chunks, num_samples))
                    chunks = None

            # takes care final batch, which is less than batch size
            if spill:
                num_samples = len(spill)
                spill = itr.preprocess(spill)
                spill = itr.create_tensors(
                    spill,
                    cat_names=self.cat_cols,
                    cont_names=self.cont_cols,
                    label_names=self.label_cols,
                )
                self.q_out.put((spill, num_samples))


    # For when an iterator is stopped before iteration is complete.
    def stop(self):
        self._stop_event.set()
        # TODO: should we be clearing? I can imagine a world where
        # you want the thread to stop but still want to grab
        # data out of the buffer
        self.q_out.queue.clear()

    def get_batch_div_chunk(self, chunks):
        spill_idx = int(chunks.shape[0] / self.batch_size) * self.batch_size
        spill = cudf.DataFrame(chunks.iloc[spill_idx:])
        chunks = cudf.DataFrame(chunks.iloc[:spill_idx])
        if not chunks.empty:
            chunks.reset_index(drop=True, inplace=True)
        if not spill.empty:
            spill.reset_index(drop=True, inplace=True)
        return chunks, spill


class AsyncIterator:
    """
        This class serves as the iterator class for the AsyncTensorBatchDatasetItr.
        This will control iteration and allow for clean up after iteration is complete.
        Without requiring the destruction of the Parent class.
        Parameters:
        dataset: NVTabular dataset
        cats: [str], the list of categorical columns in the dataset
        conts: [str], the list of continuous columns in the dataset
        labels: [str], the list of label columns in the dataset
        batch_size: int, the size of each batch to supply to the model
        shuffle: bool, enable/disable shuffling of dataset
        target: the target library that will use the tensor transformed data
                currently supported: torch
        devices: [int], list represents all avialable GPU IDs
    """

    def __init__(
        self,
        itrs,
        cats=None,
        conts=None,
        labels=None,
        batch_size=1,
        shuffle=False,
        devices=None,
        workflows=None,
    ):
        self.itrs = itrs
        self.shuffle = shuffle
        self.buff = ChunkQueue(
            batch_size=batch_size,
            cat_cols=cats,
            cont_cols=conts,
            label_cols=labels,
            shuffle=shuffle,
        )

    def __iter__(self):
        if self.shuffle:
            cp.random.shuffle(self.itrs[0].indices)

        threads = []
        for dev, itr in enumerate(self.itrs):
            t = threading.Thread(target=self.buff.load_chunks, args=(dev, itr))
            t.daemon = True
            t.start()
            threads.append(t)

        while (any([t.is_alive() for t in threads]) or not self.buff.empty):
            chunk, num_samples = self.buff.get()
            # chunk = self.itrs[0].create_tensors(
            #     chunk,
            #     self.buff.cat_cols,
            #     self.buff.cont_cols,
            #     self.buff.label_cols
            # )
            for idx in range(_num_steps(num_samples, self.buff.batch_size)):
                # TODO: how will this slicing look once we have multi-hots?
                slc = slice(idx*self.buff.batch_size, (idx+1)*self.buff.batch_size)
                outputs = []
                for t in chunk:
                    if isinstance(t, dict):
                        outputs.append({name: x[slc] for name, x in t.items()})
                    elif isinstance(t, list):
                        outputs.append([x[slc] for x in t])
                    elif t is not None:
                        outputs.append(t[slc])
                    else:
                        outputs.append(t)
                yield outputs

            chunk = None

    def __del__(self):
        self.buff.stop()


class TensorBatchDatasetItr:
    """
        Base class for all dataset to tensor iterators.
        Takes input of an NVTabular dataset
        and supplies user defined size chunks.

        Parameters
        dataset: NVTabular dataset
        shuffle: bool, specifying whether to shuffle the partitions
                 and shuffle chunks before creating tensor batches
        device: int, represents targeted GPU id
        total_devs: [int], list represents all avialable GPU IDs

    """

    def __init__(
        self, dataset, shuffle=None, workflows=None, device=0, total_devs=1, indices=None, **kwargs
    ):
        self.data = dataset
        if indices is None:
            indices = cp.arange(dataset.to_ddf().npartitions)
        self.indices = indices
        self.workflows = workflows or []

        self.device = device
        self.total_devs = total_devs

    def __iter__(self):
        indices = self.gather_indices()
        yield from self.data.to_iter(indices=indices)

    def __len__(self):
        return int(self.data.num_rows / len(self.total_devs))

    def preprocess(self, x):
        for workflow in self.workflows:
            x = workflow.apply_ops(x)
        return x

    def gather_indices(self):
        per_worker = int(len(self.indices) // len(self.total_devs)) + 1
        worker_id = self.total_devs.index(self.device)
        start = worker_id * per_worker
        return self.indices[start : start + per_worker]

    def to_dlpack(self, gdf):
        return gdf.to_dlpack()

    def device_ctx(self, dev):
        """
        This function is designed to return a context for a target device. This
        should be an integer signifying the target GPU's identifier. Currently
        this is dependent on the targeted framework. Need method exposed via
        rapids or cudf to factor this api call out.

        Parameters
        device: int, the target GPU's id
        """
        raise NotImplementedError()

    def create_tensors(self, gdf, cat_names=None, cont_names=None, label_names=None):
        raise NotImplementedError()


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


class DataLoader:
    def __init__(
            self,
            dataset,
            cat_names,
            cont_names,
            label_names,
            batch_size,
            shuffle,
            workflows=None,
            devices=None
    ):
        indices = cp.arange(dataset.to_ddf().npartitions)
        devices = devices or [0]
        workflows = workflows  or []
        _validate_workflows(
            workflows, cat_names, cont_names, label_names)

        itrs = []
        for dev in devices:
            itrs.append(self._itr_cls(
                dataset,
                indices=indices,
                device=dev,
                total_devs=devices,
                workflows=workflows
            )
        )
        self.itr = AsyncIterator(
            itrs,
            cats=cat_names,
            conts=cont_names,
            labels=label_names,
            batch_size=batch_size,
            shuffle=shuffle,
            devices=devices
        )

        self.cat_names = cat_names
        self.cont_names = cont_names
        self.label_names = label_names
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return _num_steps(self.itr.itrs[0].data.num_rows, self.batch_size)

    def __iter__(self):
        return iter(self.itr)

    def map(self, workflow):
        # TODO: this is a bit ugly, how can we clean it up?
        # maybe think about doing some class consolidation
        workflows = self.itr.itrs[0].workflows + [workflow]
        workflows = _validate_workflows(
            workflows, self.cat_names, self.cont_names, self.label_names
        )
        for itr in self.itr.itrs:
            itr.workflows = workflows
        # TODO: also need to update self.cat_names, cont_names, label_names
        # and their values in self.itr (and its buffer)
        # see point above about class consolidation
