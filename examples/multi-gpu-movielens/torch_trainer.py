import argparse
import glob
import os
from time import time

import cupy
import torch

import nvtabular as nvt
from nvtabular.framework_utils.torch.models import Model
from nvtabular.framework_utils.torch.utils import process_epoch
from nvtabular.loader.torch import DLDataLoader, TorchAsyncItr

# Horovod must be the last import to avoid conflicts
import horovod.torch as hvd  # noqa: E402, isort:skip


parser = argparse.ArgumentParser(description="Train a multi-gpu model with Torch and Horovod")
parser.add_argument("--dir_in", default=None, help="Input directory")
parser.add_argument("--batch_size", default=None, help="Batch size")
parser.add_argument("--cats", default=None, help="Categorical columns")
parser.add_argument("--cats_mh", default=None, help="Categorical multihot columns")
parser.add_argument("--conts", default=None, help="Continuous columns")
parser.add_argument("--labels", default=None, help="Label columns")
parser.add_argument("--epochs", default=1, help="Training epochs")
args = parser.parse_args()

hvd.init()

gpu_to_use = hvd.local_rank()

if torch.cuda.is_available():
    torch.cuda.set_device(gpu_to_use)


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

    max_rand = torch.iinfo(torch.int).max // hvd.size()

    # Generate a seed fragment
    seed_fragment = cupy.random.randint(0, max_rand)

    # Aggregate seed fragments from all Horovod workers
    seed_tensor = torch.tensor(seed_fragment)  # pylint: disable=not-callable
    reduced_seed = hvd.allreduce(seed_tensor, name="shuffle_seed", op=hvd.mpi_ops.Sum)

    return reduced_seed % max_rand


train_dataset = TorchAsyncItr(
    nvt.Dataset(TRAIN_PATHS),
    batch_size=BATCH_SIZE,
    cats=CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS,
    conts=NUMERIC_COLUMNS,
    labels=["rating"],
    device=gpu_to_use,
    global_size=hvd.size(),
    global_rank=hvd.rank(),
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

lr_scaler = hvd.size()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01 * lr_scaler)

hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

for epoch in range(args.epochs):
    start = time()
    print(f"Training epoch {epoch}")
    train_loss, y_pred, y = process_epoch(
        train_loader,
        model,
        train=True,
        optimizer=optimizer,
    )
    hvd.join(gpu_to_use)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    print(f"Epoch {epoch:02d}. Train loss: {train_loss:.4f}.")
    hvd.join(gpu_to_use)
    t_final = time() - start
    total_rows = train_dataset.num_rows_processed
    print(
        f"run_time: {t_final} - rows: {total_rows} - "
        f"epochs: {epoch} - dl_thru: {total_rows / t_final}"
    )


hvd.join(gpu_to_use)
if hvd.local_rank() == 0:
    print("Training complete")
