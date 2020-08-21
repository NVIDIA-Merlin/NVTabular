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

import pytest

import nvtabular as nvt
import nvtabular.io
import nvtabular.ops as ops

# If tensorflow isn't installed skip these tests. Note that the
# tf_dataloader import needs to happen after this line
tf_dataloader = pytest.importorskip("nvtabular.tf_dataloader")


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
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

    processor = nvt.Workflow(cat_names=cat_names, cont_names=cont_names, label_name=label_name)
    processor.add_feature([ops.FillMedian()])
    processor.add_preprocess(ops.Normalize())
    processor.add_preprocess(ops.Categorify())
    processor.finalize()

    data_itr = tf_dataloader.KerasSequenceDataset(
        paths if use_paths else dataset,
        columns=columns,
        batch_size=batch_size,
        buffer_size=gpu_memory_frac,
        label_name=label_name[0],
        engine=engine,
        shuffle=False,
    )
    processor.update_stats(dataset)
    data_itr.map(processor)

    rows = 0
    for idx in range(len(data_itr)):
        X, y = next(data_itr)

        # first elements to check epoch-to-epoch consistency
        if idx == 0:
            X0, y0 = X, y

        # check that we have at most batch_size elements
        num_samples = y.shape[0]
        assert num_samples <= batch_size

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

    # check start of next epoch to ensure consistency
    X, y = next(data_itr)
    assert (y.numpy() == y0.numpy()).all()
    for column, x in X.items():
        x0 = X0.pop(column)
        assert (x.numpy() == x0.numpy()).all()
    assert len(X0) == 0

    # accounts for incomplete batches at the end of chunks
    # that dont necesssarily have the full batch_size
    assert (idx + 1) * batch_size >= rows
    assert rows == (60 * 24 * 3 + 1)
