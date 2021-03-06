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

import cudf
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

import nvtabular as nvt
from nvtabular import ops as ops

tf = pytest.importorskip("tensorflow")
# If tensorflow isn't installed skip these tests. Note that the
# tf_dataloader import needs to happen after this line
tf_dataloader = pytest.importorskip("nvtabular.loader.tensorflow")


def test_tf_catname_ordering(tmpdir):
    df = cudf.DataFrame(
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


# TODO: include use_columns option
# TODO: include parts_per_chunk test
@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.06])
@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("batch_size", [1, 10, 100])
@pytest.mark.parametrize("use_paths", [True, False])
def test_tf_gpu_dl(tmpdir, paths, use_paths, dataset, batch_size, gpu_memory_frac, engine):
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
            except ValueError:
                raise AssertionError
            assert x.shape[0] == num_samples
        assert len(these_cols) == 0
        rows += num_samples

    assert (idx + 1) * batch_size >= rows
    assert rows == (60 * 24 * 3 + 1)

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
    df = cudf.DataFrame(data)
    cat_names = ["Authors", "Reviewers", "Engaging User"]
    cont_names = ["Embedding"]
    label_name = ["Post"]

    cats = cat_names >> ops.HashBucket(num_buckets=10)
    workflow = nvt.Workflow(cats + cont_names + label_name)

    data_itr = tf_dataloader.KerasSequenceLoader(
        workflow.transform(nvt.Dataset(df)),
        cat_names=cat_names,
        cont_names=cont_names,
        label_names=label_name,
        batch_size=batch_size,
        shuffle=False,
    )

    idx = 0
    for X, y in data_itr:
        assert len(X) == 7
        n_samples = y.shape[0]

        for mh_name in ["Authors", "Reviewers", "Embedding"]:
            for postfix in ["__nnzs", "__values"]:
                assert (mh_name + postfix) in X
                array = X[mh_name + postfix].numpy()[:, 0]

                if postfix == "__nnzs":
                    if mh_name == "Embedding":
                        assert (array == 3).all()
                    else:
                        lens = [
                            len(x)
                            for x in data[mh_name][idx * batch_size : idx * batch_size + n_samples]
                        ]
                        assert (array == np.array(lens)).all()
                else:
                    if mh_name == "Embedding":
                        assert len(array) == (n_samples * 3)
                    else:
                        assert len(array) == sum(lens)
        idx += 1
    assert idx == (3 // batch_size + 1)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_validater(tmpdir, batch_size):
    n_samples = 9
    rand = np.random.RandomState(0)

    gdf = cudf.DataFrame({"a": rand.randn(n_samples), "label": rand.randint(2, size=n_samples)})

    dataloader = tf_dataloader.KerasSequenceLoader(
        nvt.Dataset(gdf),
        batch_size=batch_size,
        cat_names=[],
        cont_names=["a"],
        label_names=["label"],
        shuffle=False,
    )

    input = tf.keras.Input(name="a", dtype=tf.float32, shape=(1,))
    x = tf.keras.layers.Dense(128, "relu")(input)
    x = tf.keras.layers.Dense(1, activation="softmax")(x)

    model = tf.keras.Model(inputs=input, outputs=x)
    model.compile("sgd", "binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC()])

    validater = tf_dataloader.KerasSequenceValidater(dataloader)
    model.fit(dataloader, epochs=2, verbose=0, callbacks=[validater])

    predictions, labels = [], []
    for X, y_true in dataloader:
        y_pred = model(X)
        labels.extend(y_true.numpy()[:, 0])
        predictions.extend(y_pred.numpy()[:, 0])
    predictions = np.array(predictions)
    labels = np.array(labels)

    logs = {}
    validater.on_epoch_end(0, logs)
    auc_key = [i for i in logs.keys() if i.startswith("val_auc")][0]

    true_accuracy = (labels == (predictions > 0.5)).mean()
    estimated_accuracy = logs["val_accuracy"]
    assert np.isclose(true_accuracy, estimated_accuracy, rtol=1e-6)

    true_auc = roc_auc_score(labels, predictions)
    estimated_auc = logs[auc_key]
    assert np.isclose(true_auc, estimated_auc, rtol=1e-6)
