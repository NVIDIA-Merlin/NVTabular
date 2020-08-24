import argparse
import logging
import os
import sys
import time

sys.path.insert(1, "../")


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("in_dir", help="directory with dataset files inside")
    parser.add_argument("in_file_type", help="type of file (i.e. parquet, csv, orc)")
    parser.add_argument(
        "gpu_mem_frac", help="the amount of gpu memory to use for dataloader in fraction"
    )
    parser.add_argument("--shuffle", help="toggle shuffling", required=False, default=False)
    return parser.parse_args()


args = parse_args()
print(args)

from nvtabular.torch_dataloader import TorchAsyncItr
import nvtabular as nvt

logging.basicConfig()
logging.getLogger("nvtabular").setLevel(logging.DEBUG)

shuffle = True if args.shuffle else False
data_path = args.in_dir
train_paths = [os.path.join(data_path, x) for x in os.listdir(data_path) if x.endswith("parquet")]
print(train_paths)
# import pdb; pdb.set_trace()
train_set = nvt.Dataset(train_paths, engine="parquet", part_mem_fraction=float(args.gpu_mem_frac))
cont_names = ["I" + str(x).zfill(2) for x in range(1, 14)]
cat_names = ["C" + str(x).zfill(2) for x in range(1, 24)]
cols = ["label"] + cont_names + cat_names

results = {}
for batch_size in [2 ** i for i in range(9, 24, 1)]:
    print("Checking batch size: ", batch_size)
    num_iter = max(10 * 1000 * 1000 // batch_size, 100)  # load 10e7 samples
    # import pdb; pdb.set_trace()
    t_batch_sets = TorchAsyncItr(
        train_set,
        cats=cat_names,
        conts=cont_names,
        labels=["label"],
        batch_size=batch_size,
        devices=[1, 2, 3, 5, 6, 7],
        shuffle=shuffle,
    )

    start = time.time()
    rows = 0
    for i, data in enumerate(t_batch_sets):
        rows += data[0].size()[0]

    stop = time.time()

    throughput = rows / (stop - start)
    results[batch_size] = throughput
    print(
        "batch size: ",
        batch_size,
        ", throughput: ",
        throughput,
        "items",
        rows,
        "time",
        stop - start,
    )
