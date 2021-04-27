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

import glob
import os
from time import time

import torch
from fastai.basics import Learner
from fastai.metrics import APScoreBinary, RocAucBinary
from fastai.tabular.data import TabularDataLoaders
from fastai.tabular.model import TabularModel

import nvtabular as nvt
from nvtabular.loader.torch import DLDataLoader, TorchAsyncItr
from nvtabular.ops import get_embedding_sizes


def gen_col(batch):
    return (batch[0], batch[1], batch[2].long())


def train_pytorch(workflow, out_path, cats, conts, labels, batch_size, parts_per_chunk):
    # Set paths and dataloaders
    train_paths = glob.glob(os.path.join(out_path, "train", "*.parquet"))
    valid_paths = glob.glob(os.path.join(out_path, "valid", "*.parquet"))

    train_data = nvt.Dataset(
        train_paths, engine="parquet", part_mem_fraction=0.04 / parts_per_chunk
    )
    valid_data = nvt.Dataset(
        valid_paths, engine="parquet", part_mem_fraction=0.04 / parts_per_chunk
    )

    train_data_itrs = TorchAsyncItr(
        train_data,
        batch_size=batch_size,
        cats=cats,
        conts=conts,
        labels=labels,
        parts_per_chunk=parts_per_chunk,
    )
    valid_data_itrs = TorchAsyncItr(
        valid_data,
        batch_size=batch_size,
        cats=cats,
        conts=conts,
        labels=labels,
        parts_per_chunk=parts_per_chunk,
    )

    train_dataloader = DLDataLoader(
        train_data_itrs, collate_fn=gen_col, batch_size=None, pin_memory=False, num_workers=0
    )
    valid_dataloader = DLDataLoader(
        valid_data_itrs, collate_fn=gen_col, batch_size=None, pin_memory=False, num_workers=0
    )
    databunch = TabularDataLoaders(train_dataloader, valid_dataloader)

    embeddings = list(get_embedding_sizes(workflow).values())
    # We limit the output dimension to 16
    embeddings = [[emb[0], min(16, emb[1])] for emb in embeddings]

    model = TabularModel(emb_szs=embeddings, n_cont=len(conts), out_sz=2, layers=[512, 256]).cuda()
    learn = Learner(
        databunch,
        model,
        loss_func=torch.nn.CrossEntropyLoss(),
        metrics=[RocAucBinary(), APScoreBinary()],
    )

    learning_rate = 1.32e-2
    epochs = 1
    start = time()
    learn.fit(epochs, learning_rate)
    t_final = time() - start
    total_rows = train_data_itrs.num_rows_processed + valid_data_itrs.num_rows_processed
    print(
        f"run_time: {t_final} - rows: {total_rows} - epochs: "
        + "{epochs} - dl_thru: {total_rows / t_final}"
    )
