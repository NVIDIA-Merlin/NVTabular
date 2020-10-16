import os
import time
import glob
import torch

import argparse
import pickle

import nvtabular as nvt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from nvtabular.loader.torch import TorchAsyncItr, DLDataLoader
from nvtabular.framework_utils.torch.models import Model
from nvtabular.framework_utils.torch.utils import process_epoch

parser = argparse.ArgumentParser()
parser.add_argument("--dataloader", default="NVT")  ### NVT or PYT
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--part-size", default="1GB")
parser.add_argument("--dl-shuffle", type=bool, default=False)
parser.add_argument("--dl-parts-per-chunk", type=int, default=1)
parser.add_argument("--no", type=int, default=1)
args, _ = parser.parse_known_args()


def rmspe_func(y_pred, y):
    "Return y_pred and y to non-log space and compute RMSPE"
    y_pred, y = torch.exp(y_pred) - 1, torch.exp(y) - 1
    pct_var = (y_pred - y) / y
    return (pct_var ** 2).mean().pow(0.5)


class CustomDataset(Dataset):
    """Simple dataset class for dataloader"""

    def __init__(self, df, cats, conts, labels):
        """Initialize the CustomDataset"""
        self.cats = df[cats].astype(np.int64).values
        self.conts = df[conts].astype(np.float32).values
        self.labels = df[labels].astype(np.float32).values

    def __len__(self):
        """Return the total length of the dataset"""
        dataset_size = self.cats.shape[0]
        return dataset_size

    def __getitem__(self, idx):
        """Return the batch given the indices"""
        return self.cats[idx], self.conts[idx], self.labels[idx]


def process_epoch(
    dataloader, model, train=False, optimizer=None, loss_func=torch.nn.MSELoss(), pyt_dl=False
):
    """
    The controlling function that loads data supplied via a dataloader to a model. Can be redefined
    based on parameters.
    Parameters
    -----------
    dataloader : iterator
        Iterator that contains the dataset to be submitted to the model.
    model : torch.nn.Module
        Pytorch model to run data through.
    train : bool
        Indicate whether dataloader contains training set.
    optimizer : object
        Optimizer to run in conjunction with model.
    loss_func : function
        Loss function to use, default is MSELoss.
    """
    model.train(mode=train)
    with torch.set_grad_enabled(train):
        y_list, y_pred_list = [], []
        for x_cat, x_cont, y in iter(dataloader):
            if pyt_dl:
                x_cat, x_cont, y = x_cat.cuda(), x_cont.cuda(), torch.squeeze(y).cuda()
            y_list.append(y.detach())
            y_pred = model(x_cat, x_cont)
            y_pred_list.append(y_pred.detach())
            loss = loss_func(y_pred, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    y = torch.cat(y_list)
    y_pred = torch.cat(y_pred_list)
    epoch_loss = loss_func(y_pred, y).item()
    return epoch_loss, y_pred, y


def run_benchmark(dl_train, dl_valid, pyt_dl=False):
    model = Model(
        embedding_table_shapes=EMBEDDING_TABLE_SHAPES,
        num_continuous=len(CONTINUOUS_COLUMNS),
        emb_dropout=EMBEDDING_DROPOUT_RATE,
        layer_hidden_dims=HIDDEN_DIMS,
        layer_dropout_rates=DROPOUT_RATES,
        max_output=MAX_LOG_SALES_PREDICTION,
    ).to("cuda")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    hist_dl = []
    time_start = time.time()
    for epoch in range(EPOCHS):
        train_loss, y_pred, y = process_epoch(
            dl_train, model, train=True, optimizer=optimizer, pyt_dl=pyt_dl
        )
        train_rmspe = rmspe_func(y_pred, y)
        valid_loss, y_pred, y = process_epoch(dl_valid, model, train=False, pyt_dl=pyt_dl)
        valid_rmspe = rmspe_func(y_pred, y)
        hist_dl.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_rmspe": train_rmspe.item(),
                "valid_loss": valid_loss,
                "valid_rmspe": valid_rmspe.item(),
            }
        )
        print(
            f"Epoch {epoch:02d}. Train loss: {train_loss:.4f}. Train RMSPE: {train_rmspe:.4f}. Valid loss: {valid_loss:.4f}. Valid RMSPE: {valid_rmspe:.4f}."
        )

    time_end = time.time()
    return time_end - time_start, hist_dl


DATA_DIR = os.environ.get("OUTPUT_DATA_DIR", "./data")
CATEGORICAL_COLUMNS = [
    "Store",
    "DayOfWeek",
    "Year",
    "Month",
    "Day",
    "StateHoliday",
    "CompetitionMonthsOpen",
    "Promo2Weeks",
    "StoreType",
    "Assortment",
    "PromoInterval",
    "CompetitionOpenSinceYear",
    "Promo2SinceYear",
    "State",
    "Week",
    "Events",
    "Promo_fw",
    "Promo_bw",
    "StateHoliday_fw",
    "StateHoliday_bw",
    "SchoolHoliday_fw",
    "SchoolHoliday_bw",
]
CONTINUOUS_COLUMNS = [
    "CompetitionDistance",
    "Max_TemperatureC",
    "Mean_TemperatureC",
    "Min_TemperatureC",
    "Max_Humidity",
    "Mean_Humidity",
    "Min_Humidity",
    "Max_Wind_SpeedKm_h",
    "Mean_Wind_SpeedKm_h",
    "CloudCover",
    "trend",
    "trend_DE",
    "AfterStateHoliday",
    "BeforeStateHoliday",
    "Promo",
    "SchoolHoliday",
]
LABEL_COLUMNS = ["Sales"]
COLUMNS = CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS + LABEL_COLUMNS

PREPROCESS_DIR = os.path.join(DATA_DIR, "ross_pre")
PREPROCESS_DIR_TRAIN = os.path.join(PREPROCESS_DIR, "train")
PREPROCESS_DIR_VALID = os.path.join(PREPROCESS_DIR, "valid")

EMBEDDING_DROPOUT_RATE = 0.04
DROPOUT_RATES = [0.001, 0.01]
HIDDEN_DIMS = [1000, 500]
BATCH_SIZE = 65536
LEARNING_RATE = 0.001
EPOCHS = args.epochs
DATALOADER_TYPE = args.dataloader
PART_SIZE = args.part_size
DL_SHUFFLE = args.dl_shuffle
DL_PARTS_PER_CHUNK = args.dl_parts_per_chunk
NO = args.no

pickle_filename = (
    DATALOADER_TYPE
    + "_"
    + str(NO)
    + "_"
    + PART_SIZE
    + "_"
    + str(DL_SHUFFLE)
    + "_"
    + str(DL_PARTS_PER_CHUNK)
)

# TODO: Calculate on the fly rather than recalling from previous analysis.
MAX_SALES_IN_TRAINING_SET = 38722.0
MAX_LOG_SALES_PREDICTION = 1.2 * np.log(MAX_SALES_IN_TRAINING_SET + 1.0)

# It's possible to use defaults defined within NVTabular.
EMBEDDING_TABLE_SHAPES = {
    "Assortment": (4, 3),
    "CompetitionMonthsOpen": (26, 10),
    "CompetitionOpenSinceYear": (24, 9),
    "Day": (32, 11),
    "DayOfWeek": (8, 5),
    "Events": (23, 9),
    "Month": (13, 7),
    "Promo2SinceYear": (9, 5),
    "Promo2Weeks": (27, 10),
    "PromoInterval": (5, 4),
    "Promo_bw": (7, 5),
    "Promo_fw": (7, 5),
    "SchoolHoliday_bw": (9, 5),
    "SchoolHoliday_fw": (9, 5),
    "State": (13, 7),
    "StateHoliday": (3, 3),
    "StateHoliday_bw": (4, 3),
    "StateHoliday_fw": (4, 3),
    "Store": (1116, 81),
    "StoreType": (5, 4),
    "Week": (53, 15),
    "Year": (4, 3),
}

# Here, however, we will use fast.ai's rule for embedding sizes.
for col in EMBEDDING_TABLE_SHAPES:
    EMBEDDING_TABLE_SHAPES[col] = (
        EMBEDDING_TABLE_SHAPES[col][0],
        min(600, round(1.6 * EMBEDDING_TABLE_SHAPES[col][0] ** 0.56)),
    )

TRAIN_PATHS = sorted(glob.glob(os.path.join(PREPROCESS_DIR_TRAIN, "*.parquet")))
VALID_PATHS = sorted(glob.glob(os.path.join(PREPROCESS_DIR_VALID, "*.parquet")))

print(args)

if DATALOADER_TYPE == "PYT":
    pd_train = pd.concat([pd.read_parquet(x) for x in TRAIN_PATHS])
    pd_valid = pd.concat([pd.read_parquet(x) for x in VALID_PATHS])

    ds_train = CustomDataset(
        pd_train, cats=sorted(CATEGORICAL_COLUMNS), conts=CONTINUOUS_COLUMNS, labels=LABEL_COLUMNS
    )
    ds_valid = CustomDataset(
        pd_valid, cats=sorted(CATEGORICAL_COLUMNS), conts=CONTINUOUS_COLUMNS, labels=LABEL_COLUMNS
    )

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_valid = DataLoader(ds_valid, batch_size=BATCH_SIZE, shuffle=True)

    pyt_dl_runtime, pyt_dl_hist = run_benchmark(dl_train, dl_valid, pyt_dl=True)
    print(pyt_dl_runtime)
    print(pyt_dl_hist)
    pickle.dump(
        {
            "epochs": EPOCHS,
            "dl_type": DATALOADER_TYPE,
            "part_size": PART_SIZE,
            "dl_shuffle": DL_SHUFFLE,
            "dl_part_per_chunk": DL_PARTS_PER_CHUNK,
            "no": NO,
            "runtime": pyt_dl_runtime,
            "dl_hist": pyt_dl_hist,
        },
        open("./output/" + pickle_filename + ".pickle", "wb"),
    )

if DATALOADER_TYPE == "NVT":
    # TensorItrDataset returns a single batch of x_cat, x_cont, y.
    collate_fn = lambda x: x

    train_dataset = TorchAsyncItr(
        nvt.Dataset(TRAIN_PATHS, part_size=PART_SIZE),
        batch_size=BATCH_SIZE,
        cats=CATEGORICAL_COLUMNS,
        conts=CONTINUOUS_COLUMNS,
        labels=LABEL_COLUMNS,
        shuffle=DL_SHUFFLE,
        parts_per_chunk=DL_PARTS_PER_CHUNK,
    )
    train_loader = DLDataLoader(
        train_dataset, batch_size=None, collate_fn=collate_fn, pin_memory=False, num_workers=0
    )

    valid_dataset = TorchAsyncItr(
        nvt.Dataset(VALID_PATHS, part_size=PART_SIZE),
        batch_size=BATCH_SIZE,
        cats=CATEGORICAL_COLUMNS,
        conts=CONTINUOUS_COLUMNS,
        labels=LABEL_COLUMNS,
    )
    valid_loader = DLDataLoader(
        valid_dataset, batch_size=None, collate_fn=collate_fn, pin_memory=False, num_workers=0
    )

    nvt_dl_runtime, nvt_dl_hist = run_benchmark(train_loader, valid_loader, pyt_dl=False)
    print(nvt_dl_runtime)
    print(nvt_dl_hist)
    pickle.dump(
        {
            "epochs": EPOCHS,
            "dl_type": DATALOADER_TYPE,
            "part_size": PART_SIZE,
            "dl_shuffle": DL_SHUFFLE,
            "dl_part_per_chunk": DL_PARTS_PER_CHUNK,
            "no": NO,
            "runtime": nvt_dl_runtime,
            "dl_hist": nvt_dl_hist,
        },
        open("./output/" + pickle_filename + ".pickle", "wb"),
    )
