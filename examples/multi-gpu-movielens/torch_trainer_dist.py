import argparse
import glob
import os
from time import time

import cupy
import torch
import torch.distributed as dist
import torch.nn as nn

import nvtabular as nvt
from nvtabular.framework_utils.torch.models import Model
from nvtabular.framework_utils.torch.utils import process_epoch
from nvtabular.loader.torch import DLDataLoader, TorchAsyncItr

parser = argparse.ArgumentParser(description="Train a multi-gpu model with Torch")
parser.add_argument("--dir_in", default=None, help="Input directory")
parser.add_argument("--batch_size", default=None, help="Batch size")
parser.add_argument("--cats", default=None, help="Categorical columns")
parser.add_argument("--cats_mh", default=None, help="Categorical multihot columns")
parser.add_argument("--conts", default=None, help="Continuous columns")
parser.add_argument("--labels", default=None, help="Label columns")
parser.add_argument("--epochs", default=1, help="Training epochs")
args = parser.parse_args()

BASE_DIR = os.path.expanduser(args.dir_in or "./data/")
BATCH_SIZE = int(args.batch_size or 16384)  # Batch Size
CATEGORICAL_COLUMNS = args.cats or ["movieId", "userId"]  # Single-hot
CATEGORICAL_MH_COLUMNS = args.cats_mh or ["genres"]  # Multi-hot
NUMERIC_COLUMNS = args.conts or []

# Output from ETL-with-NVTabular
TRAIN_PATHS = sorted(glob.glob(os.path.join(BASE_DIR, "train", "*.parquet")))

proc = nvt.Workflow.load(os.path.join(BASE_DIR, "workflow/"))

EMBEDDING_TABLE_SHAPES, MH_EMBEDDING_TABLE_SHAPES = nvt.ops.get_embedding_sizes(proc)
EMBEDDING_TABLE_SHAPES.update(MH_EMBEDDING_TABLE_SHAPES)


# TensorItrDataset returns a single batch of x_cat, x_cont, y.
def collate_fn(x):
    return x


# Seed with system randomness (or a static seed)
cupy.random.seed(None)


def seed_fn():
    """
    Generate consistent dataloader shuffle seeds across workers

    Reseeds each worker's dataloader each epoch to get fresh a shuffle
    that's consistent across workers.
    """

    max_rand = torch.iinfo(torch.int).max // world_size

    # Generate a seed fragment
    seed_fragment = cupy.random.randint(0, max_rand)

    # Aggregate seed fragments from all workers
    seed_tensor = torch.tensor(seed_fragment)  # pylint: disable=not-callable
    dist.all_reduce(seed_tensor, op=dist.ReduceOp.SUM)
    return seed_tensor % max_rand


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def runner(rank, world_size):
    setup(rank, world_size)
    train_dataset = TorchAsyncItr(
        nvt.Dataset(TRAIN_PATHS),
        batch_size=BATCH_SIZE,
        cats=CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS,
        conts=NUMERIC_COLUMNS,
        labels=["rating"],
        device=rank,
        global_size=world_size,
        global_rank=rank,
        shuffle=True,
        seed_fn=seed_fn,
    )
    train_loader = DLDataLoader(
        train_dataset, batch_size=None, collate_fn=collate_fn, pin_memory=False, num_workers=0
    )

    EMBEDDING_TABLE_SHAPES_TUPLE = (
        {
            CATEGORICAL_COLUMNS[0]: EMBEDDING_TABLE_SHAPES[CATEGORICAL_COLUMNS[0]],
            CATEGORICAL_COLUMNS[1]: EMBEDDING_TABLE_SHAPES[CATEGORICAL_COLUMNS[1]],
        },
        {CATEGORICAL_MH_COLUMNS[0]: EMBEDDING_TABLE_SHAPES[CATEGORICAL_MH_COLUMNS[0]]},
    )

    model = Model(
        embedding_table_shapes=EMBEDDING_TABLE_SHAPES_TUPLE,
        num_continuous=0,
        emb_dropout=0.0,
        layer_hidden_dims=[128, 128, 128],
        layer_dropout_rates=[0.0, 0.0, 0.0],
    ).cuda()

    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], find_unused_parameters=True
    )

    lr_scaler = world_size

    # optimizer = DistributedOptimizer(torch.optim.Adam, model.parameters(), lr=0.01 * lr_scaler)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01 * lr_scaler)

    total_rows = 0
    t_final = 0
    for epoch in range(args.epochs):
        start = time()
        with model.join():
            train_loss, y_pred, y = process_epoch(
                train_loader,
                model,
                train=True,
                optimizer=optimizer,
            )
        # hvd.join(gpu_to_use)
        # hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        print(f"Epoch {epoch:02d}. Train loss: {train_loss:.4f}.")
        t_final += time() - start
        total_rows += train_dataset.num_rows_processed
    print(
        f"run_time: {t_final} - rows: {total_rows} - "
        f"epochs: {epoch} - dl_thru: {total_rows / t_final}"
    )


if __name__ == "__main__":
    world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    world_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    runner(world_rank, world_size)

    torch.cuda.synchronize(device=world_rank)
    print("Training complete")
