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
import torch
import cupy as cp
from torch.utils.dlpack import from_dlpack


from nvtabular.io import Dataset, _shuffle_gdf
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
    return combine_tensors(cats, conts, label)


def create_tensors_plain(gdf, cat_cols=None, cont_cols=None, label_cols=None):
    cats, conts, label = {}, {}, {}
    _one_df(
        gdf, cats, conts, label, cat_names=cat_cols, cont_names=cont_cols, label_names=label_cols
    )
    return combine_tensors(cats, conts, label)


def combine_tensors(cats, conts, label):
    cats_list = [cats[x] for x in sorted(cats.keys())] if cats else None
    conts_list = [conts[x] for x in sorted(conts.keys())] if conts else None
    label_list = [label[x] for x in sorted(label.keys())] if label else None

    # Change cats, conts to dim=1 for column dim=0 for df sub section
    cats = torch.stack(cats_list, dim=1) if len(cats_list) > 0 else None
    conts = torch.stack(conts_list, dim=1) if len(conts_list) > 0 else None
    label = torch.cat(label_list, dim=0) if len(label_list) > 0 else None
    return cats, conts, label


def _one_df(gdf, cats, conts, label, cat_names=None, cont_names=None, label_names=None):
    gdf_cats, gdf_conts, gdf_label = (gdf[cat_names], gdf[cont_names], gdf[label_names])
    del gdf
    if len(gdf_cats) > 0:
        _to_tensor(gdf_cats, torch.long, cats, to_cpu=False)
    if len(gdf_conts) > 0:
        _to_tensor(gdf_conts, torch.float32, conts, to_cpu=False)
    if len(gdf_label) > 0:
        _to_tensor(gdf_label, torch.float32, label, to_cpu=False)


def _get_final_cols(preproc):
    if "cols" not in preproc.columns_ctx["final"]:
        preproc.create_final_cols()
    cat_names = _get_embedding_order(preproc.columns_ctx["final"]["cols"]["categorical"])
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

    if preproc:
        cat_names, cont_names, label_names = _get_final_cols(preproc)

    _one_df(
        gdf, cats, conts, label, cat_names=cat_names, cont_names=cont_names, label_names=label_names
    )

class ChunkQueue:
    def __init__(
        self,
        num_chunks=2,
        batch_size=1,
        shuffle=False,
        cat_cols=None,
        cont_cols=None,
        label_cols=None,
    ):
        self.num_chunks = num_chunks
        self.batch_size = batch_size
        self.q_in = queue.Queue(num_chunks)
        self.q_out = queue.Queue(1)
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.label_cols = label_cols
        self.shuffle = shuffle

    def get(self):
        return self.q_out.get()

    def put(self, obj):
        # check first for "end"
        if isinstance(obj, str):
            # clear the buffer
            self.create_chunk(final=True)
            # send bug out
            self.q_out.put(obj)
            return
        self.q_in.put(obj)
        if self.q_in.full():
            self.create_chunk()

    def create_chunk(self, final=False):
        chunks = []
        for _ in range(self.q_in.qsize()):
            chunks.append(self.q_in.get())
        if not chunks:
            return
        chunks = cudf.core.reshape.concat(chunks)
        chunks.reset_index(drop=True, inplace=True)
        if not final:
            # stitching
            chunks = self.stitch_run(chunks)
            if not chunks or chunks.empty:
                return
        if self.shuffle:
            _shuffle_gdf(chunks)
        chunks = create_tensors_plain(
            chunks, cat_cols=self.cat_cols, cont_cols=self.cont_cols, label_cols=self.label_cols
        )
        # chunk tensorized
        self.q_out.put(chunks)

    def stitch_run(self, chunks):
        spill_idx = int(chunks.shape[0] / self.batch_size) * self.batch_size
        spill = cudf.DataFrame(chunks.iloc[spill_idx:])
        chunks = cudf.DataFrame(chunks.iloc[:spill_idx])
        if not chunks.empty:
            chunks.reset_index(drop=True, inplace=True)
        if not spill.empty:
            spill.reset_index(drop=True, inplace=True)
            self.q_in.put(spill)
        return chunks


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


class AsyncTensorBatchDatasetItr(torch.utils.data.IterableDataset):
    def __init__(self, dataset, cats=None, conts=None, labels=None, batch_size=1, **kwargs):
        self.batch_size = batch_size
        self.cats = cats
        self.conts = conts
        self.labels = labels
        self.itr = TorchTensorBatchDatasetItr(dataset, **kwargs)


    def __iter__(self):
        buff = ChunkQueue(
            batch_size=self.batch_size,
            cat_cols=self.cats,
            cont_cols=self.conts,
            label_cols=self.labels,
        )
        threading.Thread(target=self.load_chunk, args=(buff,)).start()
        while True:
            # self.load_chunk(buff)
            chunk = buff.get()
            if isinstance(chunk, str):
                return
            yield from TensorItr(chunk, batch_size=self.batch_size)

    def load_chunk(self, buff):
        for chunk in self.itr:
            buff.put(chunk)
        # done iterating
        buff.put("end")

    def __len__(self):
        return len(self.itr)


class TorchTensorBatchDatasetItr(torch.utils.data.IterableDataset):
    """
        For Torch Only:
        Batch Tensor dataset, takes in list of files
        and creates TorchTensorBatchFileItr for each
        path supplied, supplying user defined size chunks.

        Parameters
        -----------
        paths : list of input files that represent complete dataset
    """

    def __init__(self, dataset, shuffle=None, **kwargs):
        self.dataset = dataset
        self.indices = cp.arange(dataset.to_ddf().npartitions)
        if shuffle:
            self.indices = cp.random.shuffle(self.indices)
        

    def __iter__(self):
        indices = self.gather_indices()
        yield from self.dataset.to_iter(indices=indices)

    def gather_indices(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self.indices
        else:
            per_worker = int(math.ceil(len(self.indices)/ float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            return self.indices[start:start + per_worker]
    
    

    def __len__(self):
        return self.rows


class DLDataLoader(torch.utils.data.DataLoader):
    def __len__(self):
        return self.dataset.num_rows
