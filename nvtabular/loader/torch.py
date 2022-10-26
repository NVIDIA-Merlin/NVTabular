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
import torch

import merlin.loader.torch
from nvtabular.loader.backend import _augment_schema


class TorchAsyncItr(merlin.loader.torch.Loader):
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
        dataset.schema = _augment_schema(
            dataset.schema, cats, conts, labels, sparse_names, sparse_max, sparse_as_dense
        )

        super().__init__(
            dataset,
            batch_size,
            shuffle=shuffle,
            seed_fn=seed_fn,
            parts_per_chunk=parts_per_chunk,
            global_size=global_size,
            global_rank=global_rank,
            drop_last=drop_last,
        )


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
