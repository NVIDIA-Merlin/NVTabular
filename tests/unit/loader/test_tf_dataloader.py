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

import os

from merlin.core.dispatch import HAS_GPU, make_df
from merlin.io import Dataset

try:
    import cupy
except ImportError:
    cupy = None
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import nvtabular as nvt
from nvtabular import ops

tf = pytest.importorskip("tensorflow")
# If tensorflow isn't installed skip these tests. Note that the
# tf_dataloader import needs to happen after this line
tf_dataloader = pytest.importorskip("nvtabular.loader.tensorflow")


def test_nested_list():
    num_rows = 100
    batch_size = 12

    df = pd.DataFrame(
        {
            "data": [
                np.random.rand(np.random.randint(10) + 1, 3).tolist() for i in range(num_rows)
            ],
            "data2": [np.random.rand(np.random.randint(10) + 1).tolist() for i in range(num_rows)],
            "label": [np.random.rand() for i in range(num_rows)],
        }
    )

    train_dataset = tf_dataloader.KerasSequenceLoader(
        Dataset(df),
        cont_names=["data", "data2"],
        label_names=["label"],
        batch_size=batch_size,
        shuffle=False,
    )

    batch = next(iter(train_dataset))
    # [[1,2,3],[3,1],[...],[]]
    nested_data_col = tf.RaggedTensor.from_row_splits(
        batch[0]["data__values"], tf.cast(batch[0]["data__offsets"], tf.int32)
    ).to_tensor()
    print(nested_data_col)
    true_data_col = tf.reshape(
        tf.ragged.constant(df.iloc[:batch_size, 0].tolist()).to_tensor(), [batch_size, -1]
    )
    # [1,2,3]
    multihot_data2_col = tf.RaggedTensor.from_row_splits(
        batch[0]["data2__values"], tf.cast(batch[0]["data2__offsets"], tf.int32)
    ).to_tensor()
    true_data2_col = tf.reshape(
        tf.ragged.constant(df.iloc[:batch_size, 1].tolist()).to_tensor(), [batch_size, -1]
    )
    assert nested_data_col.shape == true_data_col.shape
    assert np.allclose(nested_data_col.numpy(), true_data_col.numpy())
    assert multihot_data2_col.shape == true_data2_col.shape
    assert np.allclose(multihot_data2_col.numpy(), true_data2_col.numpy())


def test_shuffling():
    num_rows = 10000
    batch_size = 10000

    df = pd.DataFrame({"a": np.asarray(range(num_rows)), "b": np.asarray([0] * num_rows)})

    train_dataset = tf_dataloader.KerasSequenceLoader(
        Dataset(df), cont_names=["a"], label_names=["b"], batch_size=batch_size, shuffle=True
    )

    batch = next(iter(train_dataset))

    first_batch = tf.reshape(tf.cast(batch[0]["a"].cpu(), tf.int32), (batch_size,))
    in_order = tf.range(0, batch_size, dtype=tf.int32)

    assert (first_batch != in_order).numpy().any()
    assert (tf.sort(first_batch) == in_order).numpy().all()


@pytest.mark.parametrize("batch_size", [10, 9, 8])
@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize("num_rows", [100])
def test_tf_drp_reset(tmpdir, batch_size, drop_last, num_rows):
    df = make_df(
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

    data_itr = tf_dataloader.KerasSequenceLoader(
        [path],
        cat_names=cat_names,
        cont_names=cont_names,
        batch_size=batch_size,
        label_names=label_name,
        shuffle=False,
        drop_last=drop_last,
    )

    all_len = len(data_itr) if drop_last else len(data_itr) - 1
    all_rows = 0
    for idx, (X, y) in enumerate(data_itr):
        all_rows += len(X["cat1"])
        if idx < all_len:
            assert list(X["cat1"].numpy()) == [1] * batch_size
            assert list(X["cat2"].numpy()) == [2] * batch_size
            assert list(X["cat3"].numpy()) == [3] * batch_size
            assert list(X["cont1"].numpy()) == [1.0] * batch_size
            assert list(X["cont2"].numpy()) == [2.0] * batch_size
            assert list(X["cont3"].numpy()) == [3.0] * batch_size

    if drop_last and num_rows % batch_size > 0:
        assert num_rows > all_rows
    else:
        assert num_rows == all_rows


def test_tf_catname_ordering(tmpdir):
    df = make_df(
        {
            "cat1": [1] * 100,
            "cat2": [2] * 100,
            "cat3": [3] * 100,
            "label": [0] * 100,
            "cont3": [3.0] * 100,
            "cont2": [2.0] * 100,
            "cont1": [1.0] * 100,
        }
    )
    path = os.path.join(tmpdir, "dataset.parquet")
    df.to_parquet(path)
    cat_names = ["cat3", "cat2", "cat1"]
    cont_names = ["cont3", "cont2", "cont1"]
    label_name = ["label"]

    data_itr = tf_dataloader.KerasSequenceLoader(
        [path],
        cat_names=cat_names,
        cont_names=cont_names,
        batch_size=10,
        label_names=label_name,
        shuffle=False,
    )

    for X, y in data_itr:
        assert list(X["cat1"].numpy()) == [1] * 10
        assert list(X["cat2"].numpy()) == [2] * 10
        assert list(X["cat3"].numpy()) == [3] * 10
        assert list(X["cont1"].numpy()) == [1.0] * 10
        assert list(X["cont2"].numpy()) == [2.0] * 10
        assert list(X["cont3"].numpy()) == [3.0] * 10


def test_tf_map(tmpdir):
    df = make_df(
        {
            "cat1": [1] * 100,
            "cat2": [2] * 100,
            "cat3": [3] * 100,
            "label": [0] * 100,
            "sample_weight": [1.0] * 100,
            "cont2": [2.0] * 100,
            "cont1": [1.0] * 100,
        }
    )
    path = os.path.join(tmpdir, "dataset.parquet")
    df.to_parquet(path)
    cat_names = ["cat3", "cat2", "cat1"]
    cont_names = ["sample_weight", "cont2", "cont1"]
    label_name = ["label"]

    def add_sample_weight(features, labels, sample_weight_col_name="sample_weight"):
        sample_weight = tf.cast(features.pop(sample_weight_col_name) > 0, tf.float32)

        return features, labels, sample_weight

    data_itr = tf_dataloader.KerasSequenceLoader(
        [path],
        cat_names=cat_names,
        cont_names=cont_names,
        batch_size=10,
        label_names=label_name,
        shuffle=False,
    ).map(add_sample_weight)

    for X, y, sample_weight in data_itr:
        assert list(X["cat1"].numpy()) == [1] * 10
        assert list(X["cat2"].numpy()) == [2] * 10
        assert list(X["cat3"].numpy()) == [3] * 10
        assert list(X["cont1"].numpy()) == [1.0] * 10
        assert list(X["cont2"].numpy()) == [2.0] * 10

        assert list(sample_weight.numpy()) == [1.0] * 10


# TODO: include use_columns option
# TODO: include parts_per_chunk test
@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.06])
@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("batch_size", [1, 10, 100])
@pytest.mark.parametrize("use_paths", [True, False])
@pytest.mark.parametrize("cpu_true", [False, True])
@pytest.mark.parametrize("device", ["cpu", 0])
def test_tf_gpu_dl(
    tmpdir, paths, use_paths, device, cpu_true, dataset, batch_size, gpu_memory_frac, engine
):
    cont_names = ["x", "y", "id"]
    cat_names = ["name-string"]
    label_name = ["label"]
    if engine == "parquet":
        cat_names.append("name-cat")

    columns = cont_names + cat_names

    conts = cont_names >> ops.FillMedian() >> ops.Normalize()
    cats = cat_names >> ops.Categorify()

    workflow = nvt.Workflow(conts + cats + label_name)
    workflow.fit(dataset)
    workflow.transform(dataset).to_parquet(tmpdir + "/processed")

    data_itr = tf_dataloader.KerasSequenceLoader(
        str(tmpdir + "/processed"),  # workflow.transform(dataset),
        cat_names=cat_names,
        cont_names=cont_names,
        batch_size=batch_size,
        buffer_size=gpu_memory_frac,
        label_names=label_name,
        engine=engine,
        shuffle=False,
        device=device,
        reader_kwargs={"cpu": cpu_true},
    )
    _ = tf.random.uniform((1,))

    rows = 0
    for idx in range(len(data_itr)):
        X, y = next(data_itr)

        # first elements to check epoch-to-epoch consistency
        if idx == 0:
            X0, y0 = X, y

        # check that we have at most batch_size elements
        num_samples = y.shape[0]
        if num_samples != batch_size:
            try:
                next(data_itr)
            except StopIteration:
                rows += num_samples
                continue
            else:
                raise ValueError("Batch size too small at idx {}".format(idx))

        # check that all the features in X have the
        # appropriate length and that the set of
        # their names is exactly the set of names in
        # `columns`
        these_cols = columns.copy()
        for column, x in X.items():
            try:
                these_cols.remove(column)
            except ValueError as e:
                raise AssertionError from e
            assert x.shape[0] == num_samples
        assert len(these_cols) == 0
        rows += num_samples

    assert (idx + 1) * batch_size >= rows
    row_count = (60 * 24 * 3 + 1) if HAS_GPU else (60 * 24 * 3)
    assert rows == row_count
    # if num_samples is equal to batch size,
    # we didn't exhaust the iterator and do
    # cleanup. Try that now
    if num_samples == batch_size:
        try:
            next(data_itr)
        except StopIteration:
            pass
        else:
            raise ValueError
    assert not data_itr._working
    assert data_itr._batch_itr is None

    # check start of next epoch to ensure consistency
    X, y = next(data_itr)
    assert (y.numpy() == y0.numpy()).all()

    for column, x in X.items():
        x0 = X0.pop(column)
        assert (x.numpy() == x0.numpy()).all()
    assert len(X0) == 0

    data_itr.stop()
    assert not data_itr._working
    assert data_itr._batch_itr is None


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_mh_support(tmpdir, batch_size):
    data = {
        "Authors": [["User_A"], ["User_A", "User_E"], ["User_B", "User_C"], ["User_C"]],
        "Reviewers": [
            ["User_A"],
            ["User_A", "User_E"],
            ["User_B", "User_C"],
            ["User_C"],
        ],
        "Engaging User": ["User_B", "User_B", "User_A", "User_D"],
        "Embedding": [
            [0.1, 0.2, 0.3],
            [0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8],
            [0.8, 0.4, 0.2],
        ],
        "Post": [1, 2, 3, 4],
    }
    df = make_df(data)
    cat_names = ["Authors", "Reviewers", "Engaging User"]
    cont_names = ["Embedding"]
    label_name = ["Post"]
    if HAS_GPU:
        cats = cat_names >> ops.HashBucket(num_buckets=10)
    else:
        cats = cat_names >> ops.Categorify()
    workflow = nvt.Workflow(cats + cont_names + label_name)

    data_itr = tf_dataloader.KerasSequenceLoader(
        workflow.fit_transform(nvt.Dataset(df)),
        cat_names=cat_names,
        cont_names=cont_names,
        label_names=label_name,
        batch_size=batch_size,
        shuffle=False,
    )
    idx = 0

    multihot_data = df
    if HAS_GPU:
        multihot_data = multihot_data.to_pandas()

    for X, y in data_itr:
        assert len(X) == 7
        n_samples = y.shape[0]

        for mh_name in ["Authors", "Reviewers", "Embedding"]:
            # assert (mh_name) in X
            array, offsets = X[f"{mh_name}__values"], X[f"{mh_name}__offsets"]
            offsets = offsets.numpy()
            array = array.numpy()
            m_dta = [len(x) for x in multihot_data[mh_name][idx * batch_size : idx * batch_size + n_samples]]
            lens = [0] + np.cumsum(m_dta).tolist()
            cur = np.sum(m_dta)
            assert (offsets == np.array(lens)).all()
            assert len(array) == max(lens)

        idx += 1
    assert idx == (3 // batch_size + 1)


@pytest.mark.parametrize("batch_size", [128, 256])
def test_validater(tmpdir, batch_size):
    n_samples = 10000
    rand = np.random.RandomState(0)

    gdf = make_df({"a": rand.randn(n_samples), "label": rand.randint(2, size=n_samples)})

    dataloader = tf_dataloader.KerasSequenceLoader(
        nvt.Dataset(gdf),
        batch_size=batch_size,
        cat_names=[],
        cont_names=["a"],
        label_names=["label"],
        shuffle=False,
    )

    input_ = tf.keras.Input(name="a", dtype=tf.float32, shape=(1,))
    x = tf.keras.layers.Dense(128, "relu")(input_)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=input_, outputs=x)
    model.compile("sgd", "binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC()])

    validater = tf_dataloader.KerasSequenceValidater(dataloader)
    model.fit(dataloader, epochs=1, verbose=0, callbacks=[validater])

    predictions, labels = [], []
    for X, y_true in dataloader:
        y_pred = model(X)
        labels.extend(y_true.numpy())
        predictions.extend(y_pred.numpy()[:, 0])
    predictions = np.array(predictions)
    labels = np.array(labels)

    logs = {}
    validater.on_epoch_end(0, logs)
    auc_key = [i for i in logs if i.startswith("val_auc")][0]

    true_accuracy = (labels == (predictions > 0.5)).mean()
    print(true_accuracy)
    estimated_accuracy = logs["val_accuracy"]
    print(estimated_accuracy)
    assert np.isclose(true_accuracy, estimated_accuracy, rtol=0.1)

    true_auc = roc_auc_score(labels, predictions)
    estimated_auc = logs[auc_key]
    print(true_auc)
    print(estimated_auc)
    assert np.isclose(true_auc, estimated_auc, rtol=0.1)


@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("batch_size", [1, 10, 100])
@pytest.mark.parametrize("global_rank", [0, 1])
def test_multigpu_partitioning(datasets, engine, batch_size, global_rank):
    cont_names = ["x", "y", "id"]
    cat_names = ["name-string", "name-cat"]
    label_name = ["label"]
    data_loader = tf_dataloader.KerasSequenceLoader(
        str(datasets["parquet"]),
        cat_names=cat_names,
        cont_names=cont_names,
        batch_size=batch_size,
        buffer_size=0.1,
        label_names=label_name,
        engine=engine,
        shuffle=False,
        global_size=2,
        global_rank=global_rank,
    )
    indices = data_loader._indices_for_process()
    assert indices == [global_rank]


@pytest.mark.parametrize("batch_size", [1000])
@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("device", [None, 0])
def test_dataloader_schema(tmpdir, df, dataset, batch_size, engine, device):
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

    nvt_data = nvt.Dataset(tar_paths, engine="parquet")

    data_loader = tf_dataloader.KerasSequenceLoader(
        nvt_data,
        batch_size=batch_size,
        shuffle=False,
        label_names=label_name,
    )

    batch = next(iter(data_loader))
    assert all(name in batch[0] for name in cat_names)
    assert all(name in batch[0] for name in cont_names)

    num_label_cols = batch[1].shape[1] if len(batch[1].shape) > 1 else 1
    assert num_label_cols == len(label_name)

    data_loader.stop()
