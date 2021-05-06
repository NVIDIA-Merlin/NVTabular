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
import pandas as pd
import torch
import numpy as np
from torch.utils.dlpack import from_dlpack

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
    devices : [int]
        list representing all available GPU IDs
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
        sparse_list=None,
    ):
        DataLoader.__init__(
            self,
            dataset,
            cats,
            conts,
            labels,
            batch_size,
            shuffle,
            seed_fn=seed_fn,
            parts_per_chunk=parts_per_chunk,
            device=device,
            global_size=global_size,
            global_rank=global_rank,
            drop_last=drop_last,
            sparse_list=sparse_list,
        )

    def __iter__(self):
        return DataLoader.__iter__(self)

    def _get_device_ctx(self, dev):
        return torch.cuda.device("cuda:{}".format(dev))

    def _to_tensor(self, gdf, dtype=None):
        dl_pack = gdf.to_dlpack()
        tensor = from_dlpack(dl_pack)
        return tensor.type(dtype)

    def _split_fn(self, tensor, idx, axis=0):
        return torch.split(tensor, idx, dim=axis)

    @property
    def _LONG_DTYPE(self):
        return torch.long

    @property
    def _FLOAT32_DTYPE(self):
        return torch.float32

    def _handle_tensors(self, cats, conts, labels):
        X = {}
        for tensor, names in zip([cats, conts], [self.cat_names, self.cont_names]):
            lists = {}
            if isinstance(tensor, tuple):
                tensor, lists = tensor
            names = [i for i in names if i not in lists]

            # break list tuples into two keys, with postfixes
            # TODO: better choices for naming?
            list_columns = list(lists.keys())
            for column in list_columns:
                values, nnzs = lists.pop(column)
                lists[column + "__values"] = values
                lists[column + "__nnzs"] = nnzs

            # now add in any scalar tensors
            if len(names) > 1:
                tensors = self._split_fn(tensor, len(names), axis=1)
                lists.update(zip(names, tensors))
            elif len(names) == 1:
                lists[names[0]] = tensor
            X.update(lists)

        # sparse representation change-over
        for column_name in X.keys():
            if column_name in self.sparse_list:
                X[column_name] = self._to_sparse_tensor(X[column_name])

        # TODO: use dict for labels as well?
        # would require output layers to match naming
        if len(self.label_names) > 1:
            labels = self._split_fn(labels, len(self.label_names), axis=1)
        
        return X, labels


    def _to_sparse_tensor(self, values_offset):
        """
        values_offset is either a tuple (values, offsets) or just values.
        Values is a tensor.
        This method is used to turn a tensor into its sparse representation
        """
        if isinstance(values_offset, tuple):
            values = values_offset[0].flatten()
            offsets = values_offset[1].flatten()
        else:
            values = values_offset.flatten()
            offsets = torch.from_numpy(np.array([i for i in range(values.size()[0])])).to("cuda")
        num_rows = len(offsets)

        #Appending the values length to the end of the offset vector, to be able to compute diff of the last sequence
        offsets = torch.cat([offsets, torch.LongTensor([len(values)]).to("cuda")])
        #Computing the difference between consecutive offsets, to get the sequence lengths
        diff_offsets = offsets[1:] - offsets[:-1]
        #Infering the number of cols based on the maximum sequence length
        max_seq_len = int(diff_offsets.max())
        #default_seq_features_len = 1 
        #if max_seq_len > default_seq_features_len:
        #    raise ValueError('The default sequence length has been configured to {}, but the '+\
        #                        'largest sequence in this batch have {} length'.format(self.default_seq_features_len,
        #                                                                            max_seq_len))

        #Building the indices to reconstruct the sparse tensors
        row_ids = torch.arange(len(offsets)-1).to("cuda")
        row_ids_repeated = torch.repeat_interleave(row_ids, diff_offsets)
        row_offset_repeated = torch.repeat_interleave(offsets[:-1], diff_offsets)
        col_ids = torch.arange(len(row_offset_repeated)).to("cuda") - row_offset_repeated.to("cuda")
        indices = torch.cat([row_ids_repeated.unsqueeze(-1), col_ids.unsqueeze(-1)], axis=1)

        if torch.is_floating_point(values):
            sparse_tensor_class = torch.sparse.FloatTensor
        else:
            sparse_tensor_class = torch.sparse.LongTensor

        sparse_tensor = sparse_tensor_class(indices.T, values, torch.Size([num_rows, max_seq_len]))
        return sparse_tensor


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
