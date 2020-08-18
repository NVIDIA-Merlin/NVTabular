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

import torch
from torch.utils.dlpack import from_dlpack

from nvtabular.ops import _get_embedding_order
from .backend import AsyncIterator, TensorBatchDatasetItr


class TorchTensorBatchDatasetItr(TensorBatchDatasetItr):
    """
        For PyTorch Only:
        Takes input of an NVTabular dataset
        and creates TorchTensorBatchDatasetItr
        supplying user defined size chunks.

    """

    def device_ctx(self, dev):
        return torch.cuda.device(dev)

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

    _itr_cls = TorchTensorBatchDatasetItr

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
        self.devices = devices

    def __iter__(self):
        return iter(
            AsyncIterator(
                self._itr_cls,
                batch_size=self.batch_size,
                cats=self.cats,
                conts=self.conts,
                labels=self.labels,
                shuffle=self.shuffle,
                devices=self.devices,
            )
        )

    def __len__(self):
        return math.ceil(self.data.num_rows / self.batch_size)


class DLDataLoader(torch.utils.data.DataLoader):
    """
        This class is an extension of the torch dataloader.
        It is required, to support the FastAI framework.
    """

    def __len__(self):
        return len(self.dataset)
