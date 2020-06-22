import argparse
import logging
import os
from time import time


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("gpu_id", help="gpu index to use")
    parser.add_argument("in_dir", help="directory with dataset files inside")
    parser.add_argument("out_dir", help="directory to save new larger files")
    parser.add_argument("in_file_type", help="type of file (i.e. parquet, csv, orc)")
    parser.add_argument(
        "freq_thresh", help="frequency threshold for categorical can be int or dict"
    )
    parser.add_argument("batch_size", help="type of file (i.e. parquet, csv, orc)")
    parser.add_argument("gpu_mem_frac", help="size of gpu to allot to the dataset iterator")
    parser.add_argument(
        "--shuffle", required=False, help="bool value to activate shuffling of processed dataset"
    )
    parser.add_argument("--pool", required=False, help="bool value to use a RMM pooled allocator")
    return parser.parse_args()


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

logging.basicConfig()
logging.getLogger("nvtabular").setLevel(logging.DEBUG)

# delay importing any gpu code until we've set the CUDA_VISIBLE_DEVICES env
# variable above.
import cudf
import rmm
import torch
from fastai.basic_data import DataBunch
from fastai.basic_train import Learner
from fastai.metrics import accuracy
from fastai.tabular import TabularModel

from nvtabular import Workflow
from nvtabular.io import GPUDatasetIterator, device_mem_size
from nvtabular.ops import Categorify, LogOp, Normalize, ZeroFill
from nvtabular.torch_dataloader import DLCollator, DLDataLoader, FileItrDataset


if args.pool:
    rmm.reinitialize(pool_allocator=True, initial_pool_size=0.8 * device_mem_size(kind="free"))

# Args needed GPU_id, in_dir, out_dir, in_file_type, freq_threshold, batch_size, gpu_mem_frac
# day_split
print(args)
shuffle_arg = True if args.shuffle else False
print(torch.__version__, cudf.__version__)
data_path = args.in_dir
df_valid = ""
df_train = ""
# split = 270
# fin = 332
# print('Gathering input dataset files')
# train_days = [x for x in range(split)]
# valid_days = [x for x in range(split, fin)]
# print(train_days, valid_days)

start = 0
split = 23
fin = 24

train_days = [str(x) for x in range(start, split)]
valid_days = [str(x) for x in range(split, fin)]
print(train_days, valid_days)
all_days = train_days + valid_days

train_set = [
    data_path + df_train + x
    for x in os.listdir(data_path + df_train)
    if x.startswith("day") and x.split(".")[0].split("_")[-1] in train_days
]
valid_set = [
    data_path + df_valid + x
    for x in os.listdir(data_path + df_valid)
    if x.startswith("day") and x.split(".")[0].split("_")[-1] in valid_days
]

print(train_set, valid_set)

cont_names = ["I" + str(x) for x in range(1, 14)]
cat_names = ["C" + str(x) for x in range(1, 27)]
cols = ["label"] + cont_names + cat_names
print("Creating Workflow Object")
proc = Workflow(cat_names=cat_names, cont_names=cont_names, label_name=["label"])
proc.add_feature([ZeroFill(replace=True), LogOp(replace=True)])
proc.add_preprocess(Normalize(replace=True))
if int(args.freq_thresh) == 0:
    proc.add_preprocess(Categorify(replace=True))
else:
    proc.add_preprocess(
        Categorify(replace=True, use_frequency=True, freq_threshold=int(args.freq_thresh))
    )
print("Creating Dataset Iterator")
trains_itrs = None

trains_itrs = GPUDatasetIterator(
    train_set,
    names=cols,
    engine=args.in_file_type,
    gpu_memory_frac=float(args.gpu_mem_frac),
    sep="\t",
)
valids_itrs = GPUDatasetIterator(
    valid_set,
    names=cols,
    engine=args.in_file_type,
    gpu_memory_frac=float(args.gpu_mem_frac),
    sep="\t",
)
print("Running apply")

out_train = os.path.join(args.out_dir, "train")
out_valid = os.path.join(args.out_dir, "valid")

start = time()
proc.apply(
    trains_itrs,
    apply_offline=True,
    record_stats=True,
    shuffle=shuffle_arg,
    output_path=out_train,
    num_out_files=35,
)
print(f"train preprocess time: {time() - start}")

start = time()
proc.apply(
    valids_itrs,
    apply_offline=True,
    record_stats=False,
    shuffle=shuffle_arg,
    output_path=out_valid,
    num_out_files=35,
)
print(f"valid preprocess time: {time() - start}")
print(proc.timings)

embeddings = [
    x[1]
    for x in proc.df_ops["Categorify"].get_emb_sz(
        proc.stats["categories"], proc.columns_ctx["categorical"]["base"]
    )
]
print("Creating Iterators for dataloader")
start = time()

new_train_set = [os.path.join(out_train, x) for x in os.listdir(out_train) if x.endswith("parquet")]
new_valid_set = [os.path.join(out_valid, x) for x in os.listdir(out_valid) if x.endswith("parquet")]

if args.pool:
    # free up the cudf pool here so that we don't run out of memory training the model
    rmm.reinitialize(pool_allocator=False)

t_batch_sets = [
    FileItrDataset(x, names=cols, engine="parquet", batch_size=int(args.batch_size))
    for x in new_train_set
]
v_batch_sets = [
    FileItrDataset(x, names=cols, engine="parquet", batch_size=int(args.batch_size))
    for x in new_valid_set
]

t_chain = torch.utils.data.ChainDataset(t_batch_sets)
v_chain = torch.utils.data.ChainDataset(v_batch_sets)
t_final = time() - start
print(t_final)
dlc = DLCollator(preproc=proc, apply_ops=False)
print("Creating dataloaders")
start = time()
t_data = DLDataLoader(t_chain, collate_fn=dlc.gdf_col, pin_memory=False, num_workers=0)
v_data = DLDataLoader(v_chain, collate_fn=dlc.gdf_col, pin_memory=False, num_workers=0)

databunch = DataBunch(t_data, v_data, collate_fn=dlc.gdf_col, device="cuda")
t_final = time() - start
print(t_final)
print("Creating model")
start = time()
model = TabularModel(emb_szs=embeddings, n_cont=len(cont_names), out_sz=2, layers=[512, 256])
learn = Learner(databunch, model, metrics=[accuracy])
learn.loss_func = torch.nn.CrossEntropyLoss()
t_final = time() - start
print(t_final)
print("Finding learning rate")
start = time()
learn.lr_find()
learn.recorder.plot(show_moms=True, suggestion=True)
learning_rate = 1.32e-2
epochs = 1
t_final = time() - start
print(t_final)
print("Running Training")
start = time()
learn.fit_one_cycle(epochs, learning_rate)
t_final = time() - start
print(t_final)
