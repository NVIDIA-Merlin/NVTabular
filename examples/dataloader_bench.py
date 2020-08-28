import argparse
import logging
import os
import time
from glob import glob

from tqdm import tqdm


class BatchRangeAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = map(int, values.split(","))
        setattr(namespace, self.dest, [2 ** i for i in range(*values)])


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "backend", choices=("tensorflow", "torch"), help="Which backend libary to output tensors in"
    )
    parser.add_argument("in_dir", help="directory with dataset files inside")
    parser.add_argument("in_file_type", help="type of file (i.e. parquet, csv, orc)")
    parser.add_argument(
        "gpu_mem_frac", help="the amount of gpu memory to use for dataloader in fraction"
    )
    parser.add_argument("--shuffle", help="toggle shuffling", action="store_true")
    parser.add_argument("--num_devices", help="number of GPUs to benchmark on", default=1)
    parser.add_argument(
        "--batch_range",
        help=(
            "comma separate range of powers of two to sweep batches on, "
            "e.g. '9,24' for the default"
        ),
        type=str,
        default=[2 ** i for i in range(9, 24)],
        action=BatchRangeAction,
    )
    return parser.parse_args()


def main(args):
    data_path = args.in_dir
    train_paths = glob(os.path.join(data_path, "*.parquet"))
    train_set = nvt.Dataset(
        train_paths, engine="parquet", part_mem_fraction=float(args.gpu_mem_frac)
    )
    # cont_names = ["I" + str(x).zfill(2) for x in range(1, 14)]
    # cat_names = ["C" + str(x).zfill(2) for x in range(1, 24)]

    cont_names = ["I" + str(x) for x in range(1, 14)]
    cat_names = ["C" + str(x) for x in range(1, 27)]
    label_name = "label"

    for batch_size in args.batch_range:
        if args.backend == "torch":
            kwargs = {
                "cats": cat_names,
                "conts": cont_names,
                "labels": [label_name],
                "devices": [i for i in range(args.num_devices)],
            }
        else:
            kwargs = {"cat_names": cat_names, "cont_names": cont_names, "label_names": [label_name]}
        dataset = DataLoader(train_set, batch_size=batch_size, shuffle=args.shuffle, **kwargs)

        samples_seen = 0
        start_time = time.time()
        pbar = tqdm(dataset, desc="Batch size: {}".format(batch_size))
        for X in pbar:
            if args.backend == "torch":
                num_samples = X[0].size()[0]
            else:
                num_samples = X[1][0].shape[0]

            samples_seen += num_samples
            throughput = samples_seen / (time.time() - start_time)
            pbar.set_postfix(**{"samples seen": samples_seen, "throughput": throughput})


if __name__ == "__main__":
    args = parse_args()

    if args.backend == "torch":
        from nvtabular.loader.torch import TorchAsyncItr as DataLoader
    else:
        from nvtabular.loader.tensorflow import KerasSequenceLoader as DataLoader
    import nvtabular as nvt

    logging.basicConfig()
    logging.getLogger("nvtabular").setLevel(logging.DEBUG)
    main(args)
