import glob
import os
from time import time

import torch

import nvtabular as nvt
from nvtabular.framework_utils.torch.models import Model
from nvtabular.framework_utils.torch.utils import process_epoch
from nvtabular.loader.torch import DLDataLoader, TorchAsyncItr

import horovod.torch as hvd  # noqa: isort

hvd.init()

gpu_to_use = hvd.local_rank()

if torch.cuda.is_available():
    torch.cuda.set_device(gpu_to_use)

BASE_DIR = os.path.expanduser("./data/")

BATCH_SIZE = 1024 * 32  # Batch Size
CATEGORICAL_COLUMNS = ["movieId", "userId"]  # Single-hot
CATEGORICAL_MH_COLUMNS = ["genres"]  # Multi-hot
NUMERIC_COLUMNS = []

# Output from ETL-with-NVTabular
TRAIN_PATHS = sorted(glob.glob(os.path.join(BASE_DIR, "train", "*.parquet")))
VALID_PATHS = sorted(glob.glob(os.path.join(BASE_DIR, "valid", "*.parquet")))

proc = nvt.Workflow.load(os.path.join(BASE_DIR, "workflow"))

EMBEDDING_TABLE_SHAPES = nvt.ops.get_embedding_sizes(proc)


# TensorItrDataset returns a single batch of x_cat, x_cont, y.
def collate_fn(x):
    return x


train_dataset = TorchAsyncItr(
    nvt.Dataset(TRAIN_PATHS),
    batch_size=BATCH_SIZE,
    cats=CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS,
    conts=NUMERIC_COLUMNS,
    labels=["rating"],
    devices=[gpu_to_use],
    global_size=hvd.size(),
    global_rank=hvd.rank(),
)
train_loader = DLDataLoader(
    train_dataset, batch_size=None, collate_fn=collate_fn, pin_memory=False, num_workers=0
)

valid_dataset = TorchAsyncItr(
    nvt.Dataset(VALID_PATHS),
    batch_size=BATCH_SIZE,
    cats=CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS,
    conts=NUMERIC_COLUMNS,
    labels=["rating"],
    devices=[gpu_to_use],
    global_size=hvd.size(),
    global_rank=hvd.rank(),
)
valid_loader = DLDataLoader(
    valid_dataset, batch_size=None, collate_fn=collate_fn, pin_memory=False, num_workers=0
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

for epoch in range(1):
    start = time()
    train_loss, y_pred, y = process_epoch(train_loader, model, train=True, optimizer=optimizer)
    hvd.join(gpu_to_use)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    valid_loss, y_pred, y = process_epoch(valid_loader, model, train=False)
    hvd.join(gpu_to_use)
    print(f"Epoch {epoch:02d}. Train loss: {train_loss:.4f}. Valid loss: {valid_loss:.4f}.")
    t_final = time() - start
    total_rows = train_dataset.num_rows_processed + valid_dataset.num_rows_processed
    print(
        f"run_time: {t_final} - rows: {total_rows} - "
        f"epochs: {epoch} - dl_thru: {total_rows / t_final}"
    )
