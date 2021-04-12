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
import importlib
import os
import shutil
import subprocess
import time

import cudf
import numba.cuda
import pytest
from cudf.tests.utils import assert_eq

import nvtabular as nvt
import nvtabular.tools.data_gen as datagen
from nvtabular import ops as ops
from tests.conftest import mycols_csv, mycols_pq

# If pytorch isn't installed skip these tests. Note that the
# torch_dataloader import needs to happen after this line
torch = pytest.importorskip("torch")
import nvtabular.loader.torch as torch_dataloader  # noqa isort:skip
from nvtabular.framework_utils.torch.models import Model  # noqa isort:skip
from nvtabular.framework_utils.torch.utils import process_epoch  # noqa isort:skip

GPU_DEVICE_IDS = [d.id for d in numba.cuda.gpus]


@pytest.mark.parametrize("batch_size", [10, 9, 8])
@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize("num_rows", [100])
def test_torch_drp_reset(tmpdir, batch_size, drop_last, num_rows):
    df = cudf.DataFrame(
        {
            "cat1": [1] * num_rows,
            "cat2": [2] * num_rows,
            "cat3": [3] * num_rows,
            "label": [0] * num_rows,
            "cont3": [3.0] * num_rows,
            "cont2": [2.0] * num_rows,
            "cont1": [1.0] * num_rows,
        }
    )
    path = os.path.join(tmpdir, "dataset.parquet")
    df.to_parquet(path)
    cat_names = ["cat3", "cat2", "cat1"]
    cont_names = ["cont3", "cont2", "cont1"]
    label_name = ["label"]

    data_itr = torch_dataloader.TorchAsyncItr(
        nvt.Dataset([path]),
        cats=cat_names,
        conts=cont_names,
        labels=label_name,
        batch_size=batch_size,
        drop_last=drop_last,
    )

    all_len = len(data_itr) if drop_last else len(data_itr) - 1
    all_rows = 0
    for idx, chunk in enumerate(data_itr):
        all_rows += len(chunk[0])
        if idx < all_len:
            for sub in chunk[:2]:
                sub = sub.cpu()
                assert list(sub[:, 0].numpy()) == [1] * batch_size
                assert list(sub[:, 1].numpy()) == [2] * batch_size
                assert list(sub[:, 2].numpy()) == [3] * batch_size

    if drop_last and num_rows % batch_size > 0:
        assert num_rows > all_rows
    else:
        assert num_rows == all_rows


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

    features = []
    if cont_names:
        features.append(cont_names >> ops.FillMedian() >> ops.Normalize())
    if cat_names:
        features.append(cat_names >> ops.Categorify())

    # test out https://github.com/NVIDIA/NVTabular/issues/149 making sure we can iterate over
    # empty cats/conts
    graph = sum(features, nvt.ColumnGroup(label_name))
    if not graph.columns:
        # if we don't have conts/cats/labels we're done
        return

    processor = nvt.Workflow(sum(features, nvt.ColumnGroup(label_name)))

    output_train = os.path.join(tmpdir, "train/")
    os.mkdir(output_train)

    df_out = processor.fit_transform(dataset).to_ddf().compute(scheduler="synchronous")

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


@pytest.mark.parametrize("part_mem_fraction", [0.001, 0.06])
@pytest.mark.parametrize("batch_size", [1000])
@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("device", [None, 0])
def test_gpu_dl_break(tmpdir, df, dataset, batch_size, part_mem_fraction, engine, device):
    cat_names = ["name-cat", "name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    conts = cont_names >> ops.FillMedian() >> ops.Normalize()
    cats = cat_names >> ops.Categorify()

    processor = nvt.Workflow(conts + cats + label_name)

    output_train = os.path.join(tmpdir, "train/")
    os.mkdir(output_train)

    processor.fit_transform(dataset).to_parquet(
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
        device=device,
    )
    len_dl = len(data_itr) - 1

    first_chunk = 0
    for idx, chunk in enumerate(data_itr):
        if idx == 0:
            first_chunk = len(chunk[0])
        last_chk = len(chunk[0])
        print(last_chk)
        if idx == 1:
            break
        del chunk

    assert idx < len_dl

    first_chunk_2 = 0
    for idx, chunk in enumerate(data_itr):
        if idx == 0:
            first_chunk_2 = len(chunk[0])
        del chunk
    assert idx == len_dl

    assert first_chunk == first_chunk_2


@pytest.mark.parametrize("part_mem_fraction", [0.001, 0.06])
@pytest.mark.parametrize("batch_size", [1000])
@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("device", [None, 0])
def test_gpu_dl(tmpdir, df, dataset, batch_size, part_mem_fraction, engine, device):
    cat_names = ["name-cat", "name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    conts = cont_names >> ops.FillMedian() >> ops.Normalize()
    cats = cat_names >> ops.Categorify()

    processor = nvt.Workflow(conts + cats + label_name)

    output_train = os.path.join(tmpdir, "train/")
    os.mkdir(output_train)

    processor.fit_transform(dataset).to_parquet(
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
        device=device,
    )

    columns = mycols_pq
    df_test = cudf.read_parquet(tar_paths[0])[columns]
    df_test.columns = [x for x in range(0, len(columns))]
    num_rows, num_row_groups, col_names = cudf.io.read_parquet_metadata(tar_paths[0])
    rows = 0
    # works with iterator alone, needs to test inside torch dataloader

    for idx, chunk in enumerate(data_itr):
        if device is None:
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
        if device is None:
            assert float(df_test.iloc[rows][0]) == float(chunk[0][0][0])
        rows += len(chunk[0])

    if os.path.exists(output_train):
        shutil.rmtree(output_train)


@pytest.mark.parametrize("part_mem_fraction", [0.001, 0.1])
@pytest.mark.parametrize("engine", ["parquet"])
def test_kill_dl(tmpdir, df, dataset, part_mem_fraction, engine):
    cat_names = ["name-cat", "name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    conts = cont_names >> ops.FillMedian() >> ops.Normalize()
    cats = cat_names >> ops.Categorify()

    processor = nvt.Workflow(conts + cats + label_name)

    output_train = os.path.join(tmpdir, "train/")
    os.mkdir(output_train)

    processor.fit_transform(dataset).to_parquet(
        shuffle=nvt.io.Shuffle.PER_PARTITION,
        output_path=output_train,
        out_files_per_proc=2,
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

    cats = cat_names >> ops.HashBucket(num_buckets=10)

    processor = nvt.Workflow(cats + label_name)
    df_out = processor.fit_transform(nvt.Dataset(df)).to_ddf().compute(scheduler="synchronous")

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


@pytest.mark.skip(reason="fails with cuda illegal memory access")
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

    cats = cat_names >> ops.Categorify()
    conts = cont_names >> ops.Normalize()

    processor = nvt.Workflow(cats + conts + label_name)
    df_out = processor.fit_transform(nvt.Dataset(df)).to_ddf().compute()
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


@pytest.mark.skipif(importlib.util.find_spec("horovod") is None, reason="needs horovod")
def test_horovod_multigpu(tmpdir):

    json_sample = {
        "conts": {},
        "cats": {
            "genres": {
                "dtype": None,
                "cardinality": 50,
                "min_entry_size": 1,
                "max_entry_size": 5,
                "multi_min": 2,
                "multi_max": 4,
                "multi_avg": 3,
            },
            "movieId": {
                "dtype": None,
                "cardinality": 500,
                "min_entry_size": 1,
                "max_entry_size": 5,
            },
            "userId": {"dtype": None, "cardinality": 500, "min_entry_size": 1, "max_entry_size": 5},
        },
        "labels": {"rating": {"dtype": None, "cardinality": 2}},
    }
    cols = datagen._get_cols_from_schema(json_sample)
    df_gen = datagen.DatasetGen(datagen.UniformDistro(), gpu_frac=0.0001)

    target_path = os.path.join(tmpdir, "input/")
    os.mkdir(target_path)
    df_files = df_gen.full_df_create(10000, cols, output=target_path)

    # process them
    cat_features = nvt.ColumnGroup(["userId", "movieId", "genres"]) >> nvt.ops.Categorify()
    ratings = nvt.ColumnGroup(["rating"]) >> (lambda col: (col > 3).astype("int8"))
    output = cat_features + ratings

    proc = nvt.Workflow(output)
    train_iter = nvt.Dataset(df_files, part_size="10MB")
    proc.fit(train_iter)

    target_path_train = os.path.join(tmpdir, "train/")
    os.mkdir(target_path_train)

    proc.transform(train_iter).to_parquet(output_path=target_path_train, out_files_per_proc=5)

    # add new location
    target_path = os.path.join(tmpdir, "workflow/")
    os.mkdir(target_path)
    proc.save(target_path)

    curr_path = os.path.abspath(__file__)
    repo_root = os.path.relpath(os.path.normpath(os.path.join(curr_path, "../../..")))
    hvd_example_path = os.path.join(repo_root, "examples/multi-gpu-movielens/torch_trainer.py")

    process = subprocess.Popen(
        [
            "horovodrun",
            "-np",
            "2",
            "-H",
            "localhost:2",
            "python",
            hvd_example_path,
            "--dir_in",
            f"{tmpdir}",
            "--batch_size",
            "1024",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    process.wait()
    stdout, stderr = process.communicate()

    print(str(stdout))
    print(str(stderr))
    assert "Training complete" in str(stdout)
