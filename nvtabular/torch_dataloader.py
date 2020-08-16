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
        num_parts=3,
        batch_size=None,
        shuffle=False,
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
        self.stopped = False

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
                if len(chunks) > 0:
                    chunks = itr.create_tensors(
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
                spill = itr.create_tensors(
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
        dataset=None,
        cats=None,
        conts=None,
        labels=None,
        batch_size=1,
        shuffle=False,
        library=None,
        devices=None,
    ):
        self.dataset = dataset
        self.library = library
        self.shuffle = shuffle
        self.devices = devices if devices else [0]

        self.buff = ChunkQueue(
            batch_size=batch_size,
            cat_cols=cats,
            cont_cols=conts,
            label_cols=labels,
            shuffle=shuffle,
        )

    def __iter__(self):
        for dev in self.devices:
            itr = TensorBatchDatasetItrFactory().create(
                self.dataset,
                self.library,
                shuffle=self.shuffle,
                device=dev,
                total_devs=self.devices,
            )
            t1 = threading.Thread(target=self.buff.load_chunks, args=(dev, itr))
            t1.daemon = True
            t1.start()
        ends = []
        while True:
            chunk = self.buff.get()
            if isinstance(chunk, str):
                ends.append(chunk)
                if len(ends) == len(self.devices):
                    return
            else:
                yield from TensorItr(chunk, batch_size=self.buff.batch_size)
            chunk = None

    def __del__(self):
        self.buff.stop()


class TorchAsyncItr(torch.utils.data.IterableDataset):
    """
        This class, creates batches of, a user defined size, tensor
        represenation of the data supplied. The data input requires an
        NVTabular dataset. Handles spillover to ensure all batches are
        the specified size until the final batch.

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
        dataset,
        cats=None,
        conts=None,
        labels=None,
        batch_size=1,
        shuffle=False,
        target="torch",
        devices=None,
    ):
        self.batch_size = batch_size
        self.cats = cats
        self.conts = conts
        self.labels = labels
        self.shuffle = shuffle
        self.data = dataset
        self.target = target
        self.devices = devices

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
                devices=self.devices,
            )
        )

    def __len__(self):
        return math.ceil(self.data.num_rows / self.batch_size)


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

    def __init__(self, dataset, shuffle=None, device=0, total_devs=1, **kwargs):
        self.data = dataset
        self.indices = cp.arange(dataset.to_ddf().npartitions)
        #         if shuffle:
        #             cp.random.shuffle(self.indices)
        self.device = device
        self.total_devs = total_devs

    def __iter__(self):
        indices = self.gather_indices()
        yield from self.data.to_iter(indices=indices)

    def __len__(self):
        return int(self.data.num_rows / len(self.total_devs))

    def gather_indices(self):
        return self.indices

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


class TensorBatchDatasetItrFactory:
    """
        This is the Factory that creates the correct iterator,
        given a target library.

        Parameters:
        dataset: NVTabular dataset
        target: str, target library you want the iterator supported - torch
        shuffle: bool, shuffle dataset
        device: int, represents targeted GPU id
        total_devs: [int], list represents all avialable GPU IDs
    """

    def create(self, dataset, target, shuffle=False, device=None, total_devs=None, **kwargs):
        if target in "torch":
            return TorchTensorBatchDatasetItr(
                dataset, shuffle=shuffle, device=device, total_devs=total_devs, **kwargs
            )
        else:
            raise ValueError(target)


class TorchTensorBatchDatasetItr(TensorBatchDatasetItr):
    """
        For PyTorch Only:
        Takes input of an NVTabular dataset
        and creates TorchTensorBatchDatasetItr
        supplying user defined size chunks.

    """

    def device_ctx(self, dev):
        return torch.cuda.device(dev)

    def gather_indices(self):
        per_worker = int(len(self.indices) // len(self.total_devs)) + 1
        worker_id = self.total_devs.index(self.device)
        start = worker_id * per_worker
        return self.indices[start : start + per_worker].tolist()

    def _to_tensor(self, gdf, dtype=None):
        if gdf.empty:
            return
        dl_pack = self.to_dlpack(gdf)
        tens = from_dlpack(dl_pack).type(dtype)
        return tens

    def create_tensors(self, gdf, cat_names=None, cont_names=None, label_names=None):
        gdf_cats, gdf_conts, gdf_label = (
            gdf[_get_embedding_order(cat_names)],
            gdf[cont_names],
            gdf[label_names],
        )
        del gdf
        cats = self._to_tensor(gdf_cats, torch.long)
        conts = self._to_tensor(gdf_conts, torch.float32)
        label = self._to_tensor(gdf_label, torch.float32)
        del gdf_cats, gdf_conts, gdf_label
        return [cats, conts, label]


class DLDataLoader(torch.utils.data.DataLoader):
    """
        This class is an extension of the torch dataloader.
        It is required, to support the FastAI framework.
    """

    def __len__(self):
        return len(self.dataset)
