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

import cudf
import torch
from torch.utils.dlpack import from_dlpack

from nvtabular.io import GPUFileIterator


class FileItrDataset(torch.utils.data.IterableDataset):
    gpu_itr = None

    def __init__(self, file, **kwargs):
        self.gpu_itr = GPUFileIterator(file, **kwargs)

    def __iter__(self):
        return self.gpu_itr.__iter__()

    def __len__(self):
        return len(self.gpu_itr)


class TensorItrDataset(torch.utils.data.IterableDataset):
    tensor_itr = None

    def __init__(self, tensors, **kwargs):
        self.tensor_itr = TensorItr(tensors, **kwargs)

    def __iter__(self):
        return self.tensor_itr.__iter__()

    def __len__(self):
        return len(self.tensor_itr)


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
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.batch_size = batch_size
        self.cur_idx = 0
        self.num_samples = tensors[0].size(0)
        if shuffle:
            self.shuffle()

        if pin_memory:
            for tensor in self.tensors:
                tensor.pin_memory()

    def __iter__(self):
        self.cur_idx = 0
        return self

    def __len__(self):
        if self.num_samples % self.batch_size == 0:
            return self.num_samples // self.batch_size
        else:
            return self.num_samples // self.batch_size + 1

    def __next__(self):
        idx = self.cur_idx * self.batch_size
        self.cur_idx += 1
        # Need to handle odd sized batches if data isn't divisible by batchsize
        if idx < self.num_samples and (idx + self.batch_size <= self.num_samples):
            tens = [tensor[idx : idx + self.batch_size] for tensor in self.tensors]
            return (tens[0], tens[1]), tens[2]
        elif idx < self.num_samples and idx + self.batch_size > self.num_samples:
            tens = [tensor[idx:] for tensor in self.tensors]
            return (tens[0], tens[1]), tens[2]
        else:
            raise StopIteration

    def shuffle(self):
        idx = torch.randperm(self.num_samples, dtype=torch.int64)
        self.tensors = [tensor[idx] for tensor in self.tensors]


def _to_tensor(gdf: cudf.DataFrame, dtype, tensor_list, to_cpu=False):
    if gdf.empty:
        return
    for column in gdf.columns:
        gdf_col = gdf[column]
        g = gdf_col.to_dlpack()
        t = from_dlpack(g).type(dtype)
        t = t.to(torch.device("cpu")) if to_cpu else t
        tensor_list[column] = (
            t if column not in tensor_list else torch.cat([tensor_list[column], t])
        )
        del g


def create_tensors(preproc, itr=None, gdf=None, apply_ops=True):
    cats, conts, label = {}, {}, {}
    if itr:
        for gdf in itr:
            process_one_df(gdf, cats, conts, label, preproc=preproc, apply_ops=apply_ops)
    elif gdf:
        process_one_df(gdf, cats, conts, label, preproc=preproc, apply_ops=apply_ops)

    cats_list = (
        [cats[x] for x in sorted(cats.keys(), key=lambda entry: entry.split("_")[0])]
        if cats
        else None
    )
    conts_list = [conts[x] for x in sorted(conts.keys())] if conts else None
    label_list = [label[x] for x in sorted(label.keys())] if label else None

    # Change cats, conts to dim=1 for column dim=0 for df sub section
    cats = torch.stack(cats_list, dim=1) if len(cats_list) > 0 else None
    conts = torch.stack(conts_list, dim=1) if len(conts_list) > 0 else None
    label = torch.cat(label_list, dim=0) if len(label_list) > 0 else None
    return cats, conts, label


def get_final_cols(preproc):
    if "cols" not in preproc.columns_ctx["final"]:
        preproc.create_final_cols()
    cat_names = sorted(
        preproc.columns_ctx["final"]["cols"]["categorical"], key=lambda entry: entry.split("_")[0],
    )
    cont_names = sorted(preproc.columns_ctx["final"]["cols"]["continuous"])
    label_name = sorted(preproc.columns_ctx["final"]["cols"]["label"])
    return cat_names, cont_names, label_name


def process_one_df(
    gdf,
    cats,
    conts,
    label,
    preproc=None,
    cat_names=None,
    cont_names=None,
    label_names=None,
    apply_ops=True,
):
    if apply_ops and preproc:
        gdf = preproc.apply_ops(gdf)

    to_cpu = False

    if preproc:
        cat_names, cont_names, label_names = get_final_cols(preproc)
        to_cpu = preproc.to_cpu

    gdf_cats, gdf_conts, gdf_label = (
        gdf[cat_names],
        gdf[cont_names],
        gdf[label_names],
    )
    del gdf
    if len(gdf_cats) > 0:
        _to_tensor(gdf_cats, torch.long, cats, to_cpu=to_cpu)
    if len(gdf_conts) > 0:
        _to_tensor(gdf_conts, torch.float32, conts, to_cpu=to_cpu)
    if len(gdf_label) > 0:
        _to_tensor(gdf_label, torch.float32, label, to_cpu=to_cpu)


class TorchTensorBatchFileItr:
    """
        For Torch Only:
        Batch Tensor dataset, takes in a file and converts to tensors
        supplying user defined size chunks.

        Parameters
        -----------
        path : path of input file
        sub_batch_size: the size of each batch to return.
        cats: categorical columns
        conts: continuous columns
        labels: label columns
        pin_memory: allows pinning of cpu memory, if used.

    """

    def __init__(
        self, path, sub_batch_size=1, cats=None, conts=None, labels=None, pin_memory=False, **kwargs
    ):
        self.apply_ops = kwargs.get("apply_ops", False)
        self.cat_cols = cats
        self.cont_cols = conts
        self.label_cols = labels
        self.itr = GPUFileIterator(path, **kwargs)
        self.batch_size = sub_batch_size
        self.num_chunks = len(self.itr.engine)

    def proc_new_chunk(self, gdf):
        cats, conts, label = {}, {}, {}
        gdf_cats, gdf_conts, gdf_label = (
            gdf[self.cat_cols],
            gdf[self.cont_cols],
            gdf[self.label_cols],
        )
        # Change cats, conts to dim=1 for column dim=0 for df sub section
        cats = from_dlpack(gdf_cats.to_dlpack()).type(torch.long)
        conts = from_dlpack(gdf_conts.to_dlpack()).type(torch.float32)
        label = from_dlpack(gdf_label.to_dlpack()).type(torch.float32)
        return cats, conts, label

    def __iter__(self):
        for chunk in self.itr:
            chunk = self.proc_new_chunk(chunk)
            for idx in range(0, len(chunk[0]), self.batch_size):
                batch = None
                if idx + self.batch_size < len(chunk[0]):
                    batch = [tensor[idx : idx + self.batch_size] for tensor in chunk]
                else:
                    batch = [tensor[idx:] for tensor in chunk]
                yield batch


class DLCollator:
    transform = None
    preproc = None
    apply_ops = True

    def __init__(self, transform=create_tensors, preproc=None, apply_ops=True):
        self.transform = transform
        self.preproc = preproc
        self.apply_ops = apply_ops

    def gdf_col(self, gdf):
        batch = self.transform(self.preproc, gdf=gdf[0], apply_ops=self.apply_ops)
        return (batch[0], batch[1]), batch[2].long()


class TorchTensorBatchDatasetItr:
    """
        For Torch Only:
        Batch Tensor dataset, takes in list of files
        and creates TorchTensorBatchFileItr for each
        path supplied, supplying user defined size chunks.

        Parameters
        -----------
        paths : list of input files that represent complete dataset
    """

    def __init__(self, paths, **kwargs):
        if isinstance(paths, str):
            paths = [paths]
        if not isinstance(paths, list):
            raise TypeError("paths must be a string or a list.")
        if len(paths) < 1:
            raise ValueError("len(paths) must be > 0.")
        self.paths = paths
        self.cur_path = None
        self.kwargs = kwargs

    def __iter__(self):
        for path in self.paths:
            self.cur_path = path
            yield from TorchTensorBatchFileItr(path, **self.kwargs)


class DLDataLoader(torch.utils.data.DataLoader):
    def __len__(self):
        return len(self.dataset)
