import math
import queue
import threading

import cudf
import cupy as cp

from nvtabular.io import _shuffle_gdf
from nvtabular.ops import _get_embedding_order
from nvtabular.workflow import BaseWorkflow


class TensorItr:
    """
        Tensor dataset, for data already in tensor format.

        Parameters
        -----------
        tensors : list of tensors
        batch_size: the size of each batch to return.
        pin_memory: allows pinning of cpu memory, if used.
        shuffle: keyword to trigger the shuffle of batch

    """

    def __init__(self, tensors, batch_size=1, pin_memory=False, shuffle=False):
        self.tensors = tensors
        self.batch_size = batch_size

        self.num_samples = self.tensors[2].size(0)
        if shuffle:
            self.shuffle()

        if pin_memory:
            for tensor in self.tensors:
                tensor.pin_memory()

    def __len__(self):
        if self.num_samples % self.batch_size == 0:
            return self.num_samples // self.batch_size
        else:
            return self.num_samples // self.batch_size + 1

    def __iter__(self):
        for idx in range(0, self.num_samples, self.batch_size):
            yield [
                tensor[idx : idx + self.batch_size] if tensor is not None else None
                for tensor in self.tensors
            ]

    def shuffle(self):
        idx = torch.randperm(self.num_samples, dtype=torch.int64)
        self.tensors = [tensor[idx] for tensor in self.tensors]


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
        num_parts=2,
        batch_size=None,
        iterator=None,
        shuffle=False,
        cat_cols=None,
        cont_cols=None,
        label_cols=None,
    ):
        self.num_parts = num_parts
        self.itr = iterator
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

    def get(self):
        return self.q_out.get()

    def batch(self):
        current = []
        for value in self.itr:
            current.append(value)
            if len(current) == self.num_parts:
                yield current
                current = []
        if len(current) > 0:
            yield current

    def load_chunks(self):
        spill = None
        for chunks in self.batch():
            if self.stopped:
                return
            if spill and not spill.empty:
                chunks.insert(0, spill)
            chunks = cudf.core.reshape.concat(chunks)
            chunks.reset_index(drop=True, inplace=True)
            chunks, spill = self.get_batch_div_chunk(chunks)
            for workflow in self.workflows:
                chunks = workflows.apply_ops(chunks)

            if self.shuffle:
                _shuffle_gdf(chunks)
            if len(chunks) > 0:
                chunks = self.itr.create_tensors(
                    chunks,
                    cat_names=self.cat_cols,
                    cont_names=self.cont_cols,
                    label_names=self.label_cols,
                )
                # chunks tensorized
                self.q_out.put(chunks)
                chunks = None
        # takes care final batch, which is less than batch size
        if spill:
            spill = self.itr.create_tensors(
                spill,
                cat_names=self.cat_cols,
                cont_names=self.cont_cols,
                label_names=self.label_cols,
            )
            self.q_out.put(spill)
            spill = None
        self.q_out.put("end")

    # For when an iterator is stopped before iteration is complete.
    def stop(self):
        self._stop_event.set()
        # TODO: do we necessarily want to clear the queue?
        # I can imagine where you want to stop the process,
        # but let it finish out whatever it has left
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
    """

    def __init__(
        self,
        itr,
        cats=None,
        conts=None,
        labels=None,
        batch_size=1,
        shuffle=False,
        library=None,
    ):
        assert isinstance(itr, TensorBatchDatasetItr)
        self.buff = ChunkQueue(
            iterator=itr,
            batch_size=batch_size,
            cat_cols=cats,
            cont_cols=conts,
            label_cols=labels,
            shuffle=shuffle,
        )

    def __iter__(self):
        t1 = threading.Thread(target=self.buff.load_chunks)
        t1.daemon = True
        t1.start()
        while True:
            chunk = self.buff.get()
            if isinstance(chunk, str):
                break
            yield from TensorItr(chunk, batch_size=self.buff.batch_size)
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

    """
    def __init__(self, dataset, shuffle=None, **kwargs):
        self.data = dataset
        self.indices = cp.arange(dataset.to_ddf().npartitions)
        if shuffle:
            self.indices = cp.random.shuffle(self.indices)

        self.workflows = []

    def __iter__(self):
        indices = self.gather_indices()
        yield from self.data.to_iter(indices=indices)

    def map(self, workflow):
        assert isinstance(workflow, BaseWorkflow)
        self.workflows.append(workflow)

    def __len__(self):
        return self.data.num_rows

    def gather_indices(self):
        return self.indices

    def to_dlpack(self, gdf):
        return gdf.to_dlpack()

    def create_tensors(self, gdf, cat_names=None, cont_names=None, label_names=None):
        raise NotImplementedError()
