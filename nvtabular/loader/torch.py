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
import pandas as pd
import torch
from torch.utils.dlpack import from_dlpack
from cudf.utils.dtypes import is_list_dtype

from nvtabular.ops import _get_embedding_order

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
    """This class, creates batches of, a user defined size, tensor
    represenation of the data supplied. The data input requires an
    NVTabular dataset. Handles spillover to ensure all batches are
    the specified size until the final batch.

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
        parts_per_chunk=1,
        devices=None,
    ):
        DataLoader.__init__(
            self,
            dataset,
            cats,
            conts,
            labels,
            batch_size,
            shuffle,
            parts_per_chunk=parts_per_chunk,
            workflows=None,
            devices=devices,
        )

    def __iter__(self):
        return DataLoader.__iter__(self)

    def _get_device_ctx(self, dev):
        return torch.cuda.device("cuda:{}".format(dev))

    def pull_list_cols(self, gdf):
        lists = []
        reg = []
        for col in gdf.columns:
            if is_list_dtype(gdf[col]):
                lists.append(col)
            else:
                reg.append(col)
        return reg, lists

    def _to_tensor(self, gdf, dtype=None):
        tens = None
        if gdf.empty:
            return
        reg, lists = self.pull_list_cols(gdf)
        if reg:
            dl_pack = gdf[reg].to_dlpack()
            # keep next two lines separated, hurts perf, casts incorrectly
            tens = from_dlpack(dl_pack)
            tens = tens.type(dtype)
        if lists:
            list_tens = self._list_dtype_tensor(gdf, lists, dtype)
            tens = tens, list_tens
        return tens

    def _list_dtype_tensor(self, gdf, cols, dtype):
        # return a dictionary with col_name: (leaves, offsets)
        res = {}
        for col in cols:
            leaves = from_dlpack(gdf[col].list.leaves.to_dlpack())
            leaves = leaves.type(dtype)
            offsets = torch.Tensor(gdf[col]._column.offsets.values)
            res[col] = leaves, offsets
        return res

    # TODO: do we need casting or can we replace this with
    # parent class version?
    def _create_tensors(self, gdf):
        gdf_cats, gdf_conts, gdf_label = (
            gdf[_get_embedding_order(self.cat_names)],
            gdf[self.cont_names],
            gdf[self.label_names],
        )
        del gdf
        cats = self._to_tensor(gdf_cats, torch.long)
        conts = self._to_tensor(gdf_conts, torch.float32)
        label = self._to_tensor(gdf_label, torch.float32)
        del gdf_cats, gdf_conts, gdf_label
        return cats, conts, label

    def _create_batch(self, tensor, num_samples):
        tens, tensor_dict = None, None
        if type(tensor) is tuple:
            #  cat type with mh dictionary
            tensor, tensor_dict = tensor
        idx = self._get_segment_lengths(num_samples)
        if tensor is not None:
            tens = torch.split(tensor, idx)
        else:
            tens = [[]] * num_samples
        if tensor_dict:
            tens = zip(tens, self._split_lists(tensor_dict, idx))
            tens = [self._handle_dict_tens(*ten) for ten in tens]
        return tens

    def _split_lists(self, tensor_dict, idx):
        new_dict_list = []
        for col in tensor_dict.keys():
            per_col_list = []
            dl_leaves, dl_offsets = tensor_dict[col]
            # split offsets first then split leaves on offset splits
            dl_offsets_split = torch.split(dl_offsets[1:], idx)
            prev_final_offset = 0
            for x in dl_offsets_split:
                new_dict = {}
                # add previous last index as first index in new "batch"
                dl_leaves_split = dl_leaves[prev_final_offset : int(x[-1])]
                new_offsets = torch.cat([torch.tensor([0]), x - prev_final_offset], 0)
                prev_final_offset = int(x[-1])
                new_dict[col] = dl_leaves_split, new_offsets
                per_col_list.append(new_dict)
            new_dict_list.append(per_col_list)
        zip_up = zip(*new_dict_list)
        return [self._handle_dict_tens(*tens) for tens in zip_up]

    def _handle_dict_tens(self, *tens):
        return tens


class DLDataLoader(torch.utils.data.DataLoader):
    """
    This class is an extension of the torch dataloader.
    It is required, to support the FastAI framework.
    """

    def __len__(self):
        return len(self.dataset)
