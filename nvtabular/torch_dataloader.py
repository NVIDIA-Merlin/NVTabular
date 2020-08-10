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
import math
import queue
import threading

import cudf
import cupy as cp
import torch
from torch.utils.dlpack import from_dlpack

from nvtabular.io import _shuffle_gdf
from nvtabular.ops import _get_embedding_order


class TensorItr:
    """
        Tensor dataset, for data already in tensor format.
        (see preproc::ds_to_tensor)

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

        self.num_samples = self.tensors[0].size(0)
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
            tens = [tensor[idx : idx + self.batch_size] for tensor in self.tensors]
            yield tens[0], tens[1], tens[2]
            del tens

    def shuffle(self):
        idx = torch.randperm(self.num_samples, dtype=torch.int64)
        self.tensors = [tensor[idx] for tensor in self.tensors]


class ChunkQueue:
    def __init__(
        self,
        num_chunks=2,
        batch_size=None,
        iterator=None,
        shuffle=False,
        cat_cols=None,
        cont_cols=None,
        label_cols=None,
    ):
        self.num_chunks = num_chunks
        self.itr = iterator
        self.batch_size = batch_size
        self.q_out = queue.Queue(1)
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.label_cols = label_cols
        self.shuffle = shuffle
        self.stopped = False

    def get(self):
        return self.q_out.get()

    def batch(self):
        current = []
        for value in self.itr:
            current.append(value)
            if len(current) == self.num_chunks:
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
        self.stopped = True
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
    This class serves as the iterator class for the AsyncTensorBatchDatasetItr. This will control
    iteration and allow for clean up after iteration is complete. Without requiring the destruction
    of the Parent class.
    """

    def __init__(
        self,
        dataset=None,
        cats=None,
        conts=None,
        labels=None,
        batch_size=1,
        shuffle=False,
        library=None,
    ):
        itr = TensorBatchDatasetItrFactory().create(dataset, library, shuffle=shuffle)
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


class AsyncTensorBatchDatasetItr(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset,
        cats=None,
        conts=None,
        labels=None,
        batch_size=1,
        shuffle=False,
        target="torch",
    ):
        self.batch_size = batch_size
        self.cats = cats
        self.conts = conts
        self.labels = labels
        self.shuffle = shuffle
        self.data = dataset
        self.target = target

    def __iter__(self):
        return iter(
            AsyncIterator(
                dataset=self.data,
                batch_size=self.batch_size,
                cats=self.cats,
                conts=self.conts,
                labels=self.labels,
                shuffle=self.shuffle,
                library=self.target,
            )
        )

    def __len__(self):
        return math.ceil(self.data.num_rows / self.batch_size)


class TensorBatchDatasetItr:
    def __init__(self, dataset, shuffle=None, **kwargs):
        self.data = dataset
        self.indices = cp.arange(dataset.to_ddf().npartitions)
        if shuffle:
            self.indices = cp.random.shuffle(self.indices)

    def __iter__(self):
        indices = self.gather_indices()
        yield from self.data.to_iter(indices=indices)

    def __len__(self):
        return self.data.num_rows

    def gather_indices(self):
        return self.indices

    def to_dlpack(self, gdf):
        return gdf.to_dlpack()

    def to_tensor(self, dl_pack, dtype):
        raise NotImplementedError()

    def create_tensors(self, gdf, cat_names=None, cont_names=None, label_names=None):
        raise NotImplementedError()


class TensorBatchDatasetItrFactory:
    def create(self, dataset, target, shuffle=False, **kwargs):
        if target in "torch":
            return TorchTensorBatchDatasetItr(dataset, shuffle=shuffle, **kwargs)
        else:
            raise ValueError(target)


class TorchTensorBatchDatasetItr(TensorBatchDatasetItr):
    """
        For PyTorch Only:
        Batch Tensor dataset, takes in list of files
        and creates TorchTensorBatchFileItr for each
        path supplied, supplying user defined size chunks.

        Parameters
        -----------
        paths : list of input files that represent complete dataset
    """

    def gather_indices(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self.indices
        else:
            per_worker = int(len(self.indices) // worker_info.num_workers) + 1
            worker_id = worker_info.id
            start = worker_id * per_worker
            return self.indices[start : start + per_worker]

    def _to_tensor(self, gdf, dtype=None):
        dl_pack = self.to_dlpack(gdf)
        tens = from_dlpack(dl_pack).type(dtype)
        return tens, gdf.columns, dtype

    def create_tensors(self, gdf, cat_names=None, cont_names=None, label_names=None):
        gdf_cats, gdf_conts, gdf_label = (
            gdf[_get_embedding_order(cat_names)],
            gdf[cont_names],
            gdf[label_names],
        )
        del gdf
        if len(gdf_cats) > 0:
            cats = self._to_tensor(gdf_cats, torch.long)
        if len(gdf_conts) > 0:
            conts = self._to_tensor(gdf_conts, torch.float32)
        if len(gdf_label) > 0:
            label = self._to_tensor(gdf_label, torch.float32)
        del gdf_cats, gdf_conts, gdf_label
        return [cats[0], conts[0], label[0]]


class DLDataLoader(torch.utils.data.DataLoader):
    def __len__(self):
        return len(self.dataset)
