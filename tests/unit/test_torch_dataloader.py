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
import os
import shutil
import time

import cudf
import numba.cuda
import pytest
from cudf.tests.utils import assert_eq

import nvtabular as nvt
from nvtabular import ops as ops
from nvtabular.framework_utils.torch.models import Model
from nvtabular.framework_utils.torch.utils import process_epoch
from tests.conftest import mycols_csv, mycols_pq

# If pytorch isn't installed skip these tests. Note that the
# torch_dataloader import needs to happen after this line
torch = pytest.importorskip("torch")
import nvtabular.loader.torch as torch_dataloader  # noqa isort:skip

GPU_DEVICE_IDS = [d.id for d in numba.cuda.gpus]


@pytest.mark.parametrize("batch", [0, 100, 1000])
@pytest.mark.parametrize("engine", ["csv", "csv-no-header"])
def test_gpu_file_iterator_ds(df, dataset, batch, engine):
    df_itr = cudf.DataFrame()
    for data_gd in dataset.to_iter(columns=mycols_csv):
        df_itr = cudf.concat([df_itr, data_gd], axis=0) if df_itr else data_gd

    assert_eq(df_itr.reset_index(drop=True), df.reset_index(drop=True))


@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("cat_names", [["name-cat", "name-string"], []])
@pytest.mark.parametrize("cont_names", [["x", "y", "id"], []])
@pytest.mark.parametrize("label_name", [["label"], []])
def test_empty_cols(tmpdir, df, dataset, engine, cat_names, cont_names, label_name):
    # test out https://github.com/NVIDIA/NVTabular/issues/149 making sure we can iterate over
    # empty cats/conts
    # first with no continuous columns
    processor = nvt.Workflow(cat_names=cat_names, cont_names=cont_names, label_name=label_name)

    processor.add_feature([ops.FillMedian()])
    processor.add_feature(ops.Normalize())
    processor.add_feature(ops.Categorify())

    output_train = os.path.join(tmpdir, "train/")
    os.mkdir(output_train)

    processor.apply(
        dataset,
        apply_offline=True,
        record_stats=True,
        shuffle=nvt.io.Shuffle.PER_PARTITION,
        output_format=None,
    )
    df_out = processor.get_ddf().compute(scheduler="synchronous")

    data_itr = torch_dataloader.TorchAsyncItr(
        nvt.Dataset(df_out), cats=cat_names, conts=cont_names, labels=label_name, batch_size=1
    )

    for nvt_batch in data_itr:
        cats, conts, labels = nvt_batch
        if cat_names:
            assert cats.shape[-1] == len(cat_names)
        if cont_names:
            assert conts.shape[-1] == len(cont_names)
        if label_name:
            assert labels.shape[-1] == len(label_name)


@pytest.mark.parametrize("part_mem_fraction", [0.000001, 0.06])
@pytest.mark.parametrize("batch_size", [1, 10, 100])
@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("devices", [None, GPU_DEVICE_IDS[:2]])
def test_gpu_dl(tmpdir, df, dataset, batch_size, part_mem_fraction, engine, devices):
    cat_names = ["name-cat", "name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    processor = nvt.Workflow(cat_names=cat_names, cont_names=cont_names, label_name=label_name)

    processor.add_feature([ops.FillMedian()])
    processor.add_feature(ops.Normalize())
    processor.add_preprocess(ops.Categorify())

    output_train = os.path.join(tmpdir, "train/")
    os.mkdir(output_train)

    processor.apply(
        dataset,
        apply_offline=True,
        record_stats=True,
        shuffle=nvt.io.Shuffle.PER_PARTITION,
        output_path=output_train,
        out_files_per_proc=2,
    )

    tar_paths = [
        os.path.join(output_train, x) for x in os.listdir(output_train) if x.endswith("parquet")
    ]

    nvt_data = nvt.Dataset(tar_paths[0], engine="parquet", part_mem_fraction=part_mem_fraction)
    data_itr = torch_dataloader.TorchAsyncItr(
        nvt_data,
        batch_size=batch_size,
        cats=cat_names,
        conts=cont_names,
        labels=["label"],
        devices=devices,
    )

    columns = mycols_pq
    df_test = cudf.read_parquet(tar_paths[0])[columns]
    df_test.columns = [x for x in range(0, len(columns))]
    num_rows, num_row_groups, col_names = cudf.io.read_parquet_metadata(tar_paths[0])
    rows = 0
    # works with iterator alone, needs to test inside torch dataloader

    for idx, chunk in enumerate(data_itr):
        if devices is None:
            assert float(df_test.iloc[rows][0]) == float(chunk[0][0][0])
        rows += len(chunk[0])
        del chunk
    # accounts for incomplete batches at the end of chunks
    # that dont necesssarily have the full batch_size
    assert rows == num_rows

    def gen_col(batch):
        batch = batch[0]
        return batch[0], batch[1], batch[2]

    t_dl = torch_dataloader.DLDataLoader(
        data_itr, collate_fn=gen_col, pin_memory=False, num_workers=0
    )
    rows = 0
    for idx, chunk in enumerate(t_dl):
        if devices is None:
            assert float(df_test.iloc[rows][0]) == float(chunk[0][0][0])
        rows += len(chunk[0])

    if os.path.exists(output_train):
        shutil.rmtree(output_train)


@pytest.mark.parametrize("part_mem_fraction", [0.000001, 0.1])
@pytest.mark.parametrize("engine", ["parquet"])
def test_kill_dl(tmpdir, df, dataset, part_mem_fraction, engine):
    cat_names = ["name-cat", "name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    processor = nvt.Workflow(cat_names=cat_names, cont_names=cont_names, label_name=label_name)

    processor.add_feature([ops.FillMedian()])
    processor.add_feature(ops.Normalize())
    processor.add_feature(ops.Categorify())

    output_train = os.path.join(tmpdir, "train/")
    os.mkdir(output_train)

    processor.finalize()

    processor.apply(
        dataset,
        apply_offline=True,
        record_stats=True,
        shuffle=nvt.io.Shuffle.PER_PARTITION,
        output_path=output_train,
    )

    tar_paths = [
        os.path.join(output_train, x) for x in os.listdir(output_train) if x.endswith("parquet")
    ]

    nvt_data = nvt.Dataset(tar_paths[0], engine="parquet", part_mem_fraction=part_mem_fraction)

    data_itr = torch_dataloader.TorchAsyncItr(
        nvt_data, cats=cat_names, conts=cont_names, labels=["label"]
    )

    results = {}

    for batch_size in [2 ** i for i in range(9, 25, 1)]:
        print("Checking batch size: ", batch_size)
        num_iter = max(10 * 1000 * 1000 // batch_size, 100)  # load 10e7 samples
        # import pdb; pdb.set_trace()
        data_itr.batch_size = batch_size
        start = time.time()
        for i, data in enumerate(data_itr):
            if i >= num_iter:
                break
            del data

        stop = time.time()

        throughput = i * batch_size / (stop - start)
        results[batch_size] = throughput
        print(
            "batch size: ",
            batch_size,
            ", throughput: ",
            throughput,
            "items",
            i * batch_size,
            "time",
            stop - start,
        )


def test_mh_support(tmpdir):
    df = cudf.DataFrame(
        {
            "Authors": [["User_A"], ["User_A", "User_E"], ["User_B", "User_C"], ["User_C"]],
            "Reviewers": [["User_A"], ["User_A", "User_E"], ["User_B", "User_C"], ["User_C"]],
            "Engaging User": ["User_B", "User_B", "User_A", "User_D"],
            "Post": [1, 2, 3, 4],
        }
    )
    cat_names = ["Authors", "Reviewers"]  # , "Engaging User"]
    cont_names = []
    label_name = ["Post"]

    processor = nvt.Workflow(cat_names=cat_names, cont_names=cont_names, label_name=label_name)
    processor.add_preprocess(ops.HashBucket(num_buckets=10))
    processor.finalize()
    processor.apply(nvt.Dataset(df), output_format=None)
    df_out = processor.get_ddf().compute(scheduler="synchronous")

    # check to make sure that the same strings are hashed the same
    authors = df_out["Authors"].to_arrow().to_pylist()
    assert authors[0][0] == authors[1][0]  # 'User_A'
    assert authors[2][1] == authors[3][0]  # 'User_C'

    data_itr = torch_dataloader.TorchAsyncItr(
        nvt.Dataset(df_out), cats=cat_names, conts=cont_names, labels=label_name
    )
    idx = 0
    for batch in data_itr:
        idx = idx + 1
        cats, conts, labels = batch
        cats, mh = cats
        # mh is a tuple of dictionaries {Column name: (values, offsets)}
        assert len(mh) == len(cat_names)
        assert not cats
    assert idx > 0


def test_mh_model_support(tmpdir):
    df = cudf.DataFrame(
        {
            "Authors": [["User_A"], ["User_A", "User_E"], ["User_B", "User_C"], ["User_C"]],
            "Reviewers": [["User_A"], ["User_A", "User_E"], ["User_B", "User_C"], ["User_C"]],
            "Engaging User": ["User_B", "User_B", "User_A", "User_D"],
            "Null User": ["User_B", "User_B", "User_A", "User_D"],
            "Post": [1, 2, 3, 4],
            "Cont1": [0.3, 0.4, 0.5, 0.6],
            "Cont2": [0.3, 0.4, 0.5, 0.6],
            "Cat1": ["A", "B", "A", "C"],
        }
    )
    cat_names = ["Cat1", "Null User", "Authors", "Reviewers"]  # , "Engaging User"]
    cont_names = ["Cont1", "Cont2"]
    label_name = ["Post"]
    out_path = os.path.join(tmpdir, "train/")
    os.mkdir(out_path)
    processor = nvt.Workflow(cat_names=cat_names, cont_names=cont_names, label_name=label_name)
    processor.add_preprocess(ops.Normalize())
    processor.add_preprocess(ops.Categorify())
    processor.finalize()
    processor.apply(
        nvt.Dataset(df),
        record_stats=True,
    )
    df_out = processor.get_ddf().compute(scheduler="synchronous")
    data_itr = torch_dataloader.TorchAsyncItr(
        nvt.Dataset(df_out),
        cats=cat_names,
        conts=cont_names,
        labels=label_name,
        batch_size=2,
    )
    emb_sizes = nvt.ops.get_embedding_sizes(processor)
    EMBEDDING_DROPOUT_RATE = 0.04
    DROPOUT_RATES = [0.001, 0.01]
    HIDDEN_DIMS = [1000, 500]
    LEARNING_RATE = 0.001
    model = Model(
        embedding_table_shapes=emb_sizes,
        num_continuous=len(cont_names),
        emb_dropout=EMBEDDING_DROPOUT_RATE,
        layer_hidden_dims=HIDDEN_DIMS,
        layer_dropout_rates=DROPOUT_RATES,
    ).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def rmspe_func(y_pred, y):
        "Return y_pred and y to non-log space and compute RMSPE"
        y_pred, y = torch.exp(y_pred) - 1, torch.exp(y) - 1
        pct_var = (y_pred - y) / y
        return (pct_var ** 2).mean().pow(0.5)

    train_loss, y_pred, y = process_epoch(
        data_itr,
        model,
        train=True,
        optimizer=optimizer,
        # transform=batch_transform,
        amp=False,
    )
    train_rmspe = None
    train_rmspe = rmspe_func(y_pred, y)
    assert train_rmspe is not None
    assert len(y_pred) > 0
    assert len(y) > 0
