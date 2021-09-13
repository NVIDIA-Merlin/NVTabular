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

import cupy
import pytest

import nvtabular as nvt
import nvtabular.ops as ops
from nvtabular.io.dataset import Dataset
from nvtabular.loader.backend import DataLoader
from nvtabular.loader.tensorflow import KerasSequenceLoader
from nvtabular.loader.torch import TorchAsyncItr


@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("batch_size", [128])
def test_dataloader_seeding(datasets, engine, batch_size):
    cont_names = ["x", "y", "id"]
    cat_names = ["name-string", "name-cat"]
    label_name = ["label"]

    dataset = Dataset(str(datasets["parquet"]), engine=engine)

    # Define a seed function that returns the same seed on all workers
    seed_fragments = []

    def seed_fn():
        # Capturing the next random number generated allows us to check
        # that different workers have different random states when this
        # function is called
        rng_state = cupy.random.get_random_state()
        next_rand = rng_state.tomaxint(size=1)
        seed_fragments.append(next_rand)

        # But since we don't actually want to run two data loaders in
        # parallel in this test, we'll cheat and return a static seed
        # instead of combining the fragments into a new seed
        return 5678

    # Set up two dataloaders with different global ranks
    data_loader_0 = DataLoader(
        dataset,
        cat_names=cat_names,
        cont_names=cont_names,
        batch_size=batch_size,
        label_names=label_name,
        shuffle=False,
        global_size=2,
        global_rank=0,
        seed_fn=seed_fn,
    )

    data_loader_1 = DataLoader(
        dataset,
        cat_names=cat_names,
        cont_names=cont_names,
        batch_size=batch_size,
        label_names=label_name,
        shuffle=False,
        global_size=2,
        global_rank=1,
        seed_fn=seed_fn,
    )

    # Starting from the same random state, run a shuffle on each worker
    # and capture the results
    cupy.random.seed(1234)

    data_loader_0._shuffle_indices()

    dl0_rng_state = cupy.random.get_random_state()
    dl0_next_rand = dl0_rng_state.tomaxint(size=1)
    dl0_indices = data_loader_0.indices

    cupy.random.seed(1234)

    data_loader_1._shuffle_indices()

    dl1_rng_state = cupy.random.get_random_state()
    dl1_next_rand = dl1_rng_state.tomaxint(size=1)
    dl1_indices = data_loader_1.indices

    # Test that the seed function actually gets called in each data loader
    assert len(seed_fragments) == 2

    # Test that each data loader had different random state
    # when seed_fn was called
    assert seed_fragments[0] != seed_fragments[1]

    # Test that the shuffle has the same result on both workers
    # (i.e. the random seeds are the same when the shuffle happens)
    for idx, element in enumerate(dl0_indices):
        assert dl0_indices[idx] == dl1_indices[idx]

    # Test that after the shuffle each worker generates different random numbers
    # (i.e. the random seeds are different on each worker after the shuffle)
    assert dl0_next_rand != dl1_next_rand


@pytest.mark.parametrize("batch_size", [1000])
@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("device", [None, 0])
@pytest.mark.parametrize("torch_on", [False, True])
def test_dataloader_schema(tmpdir, df, dataset, batch_size, engine, device, torch_on):
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

    if torch_on:
        data_loader = TorchAsyncItr(
            nvt_data,
            batch_size=batch_size,
            shuffle=False,
            labels=label_name,
        )
    else:
        data_loader = KerasSequenceLoader(
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
