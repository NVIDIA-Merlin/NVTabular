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
import numpy as np
import pandas as pd
import torch
from torch.utils.dlpack import from_dlpack

from merlin.core.dispatch import HAS_GPU

from .backend import DataLoader


class IterDL(torch.utils.data.IterableDataset):
    def __init__(self, file_paths, batch_size=1, shuffle=False):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        for file_path in self.file_paths:
            pdf = pd.read_parquet(file_path)
            for start in range(0, pdf.shape[0], self.batch_size):
                df = pdf[start : start + self.batch_size]
                if self.shuffle:
                    df = df.sample(frac=1).reset_index(drop=True)
                yield df


class TorchAsyncItr(torch.utils.data.IterableDataset, DataLoader):
    """This class creates batches of tensor. Each batch size is specified by the user.
    The data input requires an NVTabular dataset. Handles spillover to ensure all
    batches are the specified size until the final batch.

    Parameters
    -----------
    dataset : NVTabular dataset
    cats : [str]
        the list of categorical columns in the dataset
    conts : [str]
        the list of continuous columns in the dataset
    labels : [str]
        the list of label columns in the dataset
    batch_size : int
        the size of each batch to supply to the model
    shuffle : bool
        enable/disable shuffling of dataset
    parts_per_chunk : int
        number of partitions from the iterator, an NVTabular Dataset, to concatenate into a "chunk"
    device : int
        device id of selected GPU
    sparse_list : [str]
        list with column names of columns that should be represented as sparse tensors
    sparse_max : {str: int}
        dictionary of key: column_name + value: integer representing max sequence length for column
    sparse_as_dense : bool
        bool value to activate transforming sparse tensors to dense
    """

    def __init__(
        self,
        dataset,
        cats=None,
        conts=None,
        labels=None,
        batch_size=1,
        shuffle=False,
        seed_fn=None,
        parts_per_chunk=1,
        device=None,
        global_size=None,
        global_rank=None,
        drop_last=False,
        sparse_names=None,
        sparse_max=None,
        sparse_as_dense=False,
    ):
        DataLoader.__init__(
            self,
            dataset,
            batch_size,
            shuffle,
            cat_names=cats,
            cont_names=conts,
            label_names=labels,
            seed_fn=seed_fn,
            parts_per_chunk=parts_per_chunk,
            device=device,
            global_size=global_size,
            global_rank=global_rank,
            drop_last=drop_last,
            sparse_names=sparse_names,
            sparse_max=sparse_max,
            sparse_as_dense=sparse_as_dense,
        )

    def __iter__(self):
        return DataLoader.__iter__(self)

    def _get_device_ctx(self, dev):
        if dev == "cpu":
            return torch.device("cpu")
        return torch.cuda.device("cuda:{}".format(dev))

    def _pack(self, gdf):
        if self.device == "cpu":
            return gdf
        return gdf.to_dlpack()

    def _unpack(self, dlpack):
        if self.device == "cpu":
            if (
                len(dlpack.values.shape) == 2
                and dlpack.values.shape[1] == 1
                and isinstance(dlpack.values[0], np.ndarray)
            ):
                return torch.squeeze(torch.Tensor(dlpack.values))
            return torch.Tensor(dlpack.values)
        return from_dlpack(dlpack)

    def _to_tensor(self, gdf, dtype=None):
        dl_pack = self._pack(gdf)
        tensor = self._unpack(dl_pack)
        return tensor.type(dtype)

    def _split_fn(self, tensor, idx, axis=0):
        return torch.split(tensor, idx, dim=axis)

    def _tensor_split(self, tensor, idx, axis=0):
        return torch.tensor_split(tensor, idx, axis=axis)

    @property
    def _LONG_DTYPE(self):
        return torch.long

    @property
    def _FLOAT32_DTYPE(self):
        return torch.float32

    def _pull_values_offsets(self, values_offset):
        # pull_values_offsets, return values offsets diff_offsets
        if isinstance(values_offset, tuple):
            values = values_offset[0].flatten()
            offsets = values_offset[1].flatten()
        else:
            values = values_offset.flatten()
            offsets = torch.arange(values.size()[0], device=self.device)
        num_rows = len(offsets)
        if HAS_GPU:
            offsets = torch.cat([offsets, torch.cuda.LongTensor([len(values)])])
        else:
            offsets = torch.cat([offsets, torch.LongTensor([len(values)])])
        diff_offsets = offsets[1:] - offsets[:-1]
        return values, offsets, diff_offsets, num_rows

    def _get_max_seq_len(self, diff_offsets):
        return int(diff_offsets.max())

    # Building the indices to reconstruct the sparse tensors

    def _get_indices(self, offsets, diff_offsets):
        row_ids = torch.arange(len(offsets) - 1, device=self.device)
        row_ids_repeated = torch.repeat_interleave(row_ids, diff_offsets)
        row_offset_repeated = torch.repeat_interleave(offsets[:-1], diff_offsets)
        col_ids = torch.arange(len(row_offset_repeated), device=self.device) - row_offset_repeated
        indices = torch.cat([row_ids_repeated.unsqueeze(-1), col_ids.unsqueeze(-1)], axis=1)
        return indices

    def _get_sparse_tensor(self, values, indices, num_rows, seq_limit):
        sparse_tensor = torch.sparse_coo_tensor(
            indices.T, values, torch.Size([num_rows, seq_limit]), device=self.device
        )
        if self.sparse_as_dense:
            sparse_tensor = sparse_tensor.to_dense()
        return sparse_tensor

    def _build_sparse_tensor(self, values, offsets, diff_offsets, num_rows, seq_limit):
        indices = self._get_indices(offsets, diff_offsets)
        return self._get_sparse_tensor(values, indices, num_rows, seq_limit)


class DLDataLoader(torch.utils.data.DataLoader):
    """
    This class is an extension of the torch dataloader.
    It is required to support the FastAI framework.
    """

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.dataset)
