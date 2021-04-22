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

import argparse
import os
import shutil
import time
import warnings

import rmm
from dask.distributed import Client, performance_report
from dask_cuda import LocalCUDACluster

from nvtabular import Dataset, Workflow
from nvtabular import io as nvt_io
from nvtabular import ops as ops
from nvtabular.utils import _pynvml_mem_size, device_mem_size, get_rmm_size

import gc
import glob

import nvtabular as nvt

def train_hugectr():
    import hugectr
    from mpi4py import MPI

    solver = hugectr.solver_parser_helper(
                                        vvgpu = [[0]],
                                        max_iter = 10000,
                                        max_eval_batches = 100,
                                        batchsize_eval = 2720,
                                        batchsize = 2720,
                                        display = 1000,
                                        eval_interval = 3200,
                                        snapshot = 3200,
                                        i64_input_key = True,
                                        use_mixed_precision = False,
                                        repeat_dataset = True)
    optimizer = hugectr.optimizer.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.SGD,
                                        use_mixed_precision = False)
    model = hugectr.Model(solver, optimizer)
    model.add(hugectr.Input(data_reader_type = hugectr.DataReaderType_t.Parquet,
                                source = "/raid/data/criteo/test_dask/output/train/_file_list.txt",
                                eval_source = "/raid/data/criteo/test_dask/output/valid/_file_list.txt",
                                check_type = hugectr.Check_t.Non,
                                label_dim = 1, label_name = "label",
                                dense_dim = 13, dense_name = "dense",
                                slot_size_array = [10000000, 10000000, 3014529, 400781, 11, 2209, 11869, 148, 4, 977, 15, 38713, 10000000, 10000000, 10000000, 584616, 12883, 109, 37, 17177, 7425, 20266, 4, 7085, 1535, 64],
                                data_reader_sparse_param_array = 
                                [hugectr.DataReaderSparseParam(hugectr.DataReaderSparse_t.Localized, 26, 1, 26)],
                                sparse_names = ["data1"]))
    model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.LocalizedSlotSparseEmbeddingHash, 
                                max_vocabulary_size_per_gpu = 15500000,
                                embedding_vec_size = 128,
                                combiner = 0,
                                sparse_embedding_name = "sparse_embedding1",
                                bottom_name = "data1"))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                                bottom_names = ["dense"],
                                top_names = ["fc1"],
                                num_output=512))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                                bottom_names = ["fc1"],
                                top_names = ["relu1"]))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                                bottom_names = ["relu1"],
                                top_names = ["fc2"],
                                num_output=256))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                                bottom_names = ["fc2"],
                                top_names = ["relu2"]))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                                bottom_names = ["relu2"],
                                top_names = ["fc3"],
                                num_output=128))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                                bottom_names = ["fc3"],
                                top_names = ["relu3"]))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Interaction,
                                bottom_names = ["relu3", "sparse_embedding1"],
                                top_names = ["interaction1"]))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                                bottom_names = ["interaction1"],
                                top_names = ["fc4"],
                                num_output=1024))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                                bottom_names = ["fc4"],
                                top_names = ["relu4"]))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                                bottom_names = ["relu4"],
                                top_names = ["fc5"],
                                num_output=1024))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                                bottom_names = ["fc5"],
                                top_names = ["relu5"]))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                                bottom_names = ["relu5"],
                                top_names = ["fc6"],
                                num_output=512))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                                bottom_names = ["fc6"],
                                top_names = ["relu6"]))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                                bottom_names = ["relu6"],
                                top_names = ["fc7"],
                                num_output=256))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                                bottom_names = ["fc7"],
                                top_names = ["relu7"]))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                                bottom_names = ["relu7"],
                                top_names = ["fc8"],
                                num_output=1))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,
                                bottom_names = ["fc8", "label"],
                                top_names = ["loss"]))
    model.compile()
    model.summary()
    model.fit()

def train_tensorflow():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    import tensorflow as tf

    from tensorflow.python.feature_column import feature_column_v2 as fc

    # we can control how much memory to give tensorflow with this environment variable
    # IMPORTANT: make sure you do this before you initialize TF's runtime, otherwise
    # TF will have claimed all free GPU memory
    os.environ['TF_MEMORY_ALLOCATION'] = "0.5" # fraction of free memory
    from nvtabular.loader.tensorflow import KerasSequenceLoader, KerasSequenceValidater
    from nvtabular.framework_utils.tensorflow import layers
    from tensorflow.python.feature_column import feature_column_v2 as fc

    train_dataset_tf = KerasSequenceLoader(
    TRAIN_PATHS,
    batch_size=BATCH_SIZE,
    label_names=LABEL_COLUMNS,
    cat_names=CATEGORICAL_COLUMNS,
    cont_names=CONTINUOUS_COLUMNS,
    engine='parquet',
    shuffle=True,
    buffer_size=0.06,
    parts_per_chunk=1
    )

    valid_dataset_tf = KerasSequenceLoader(
        VALID_PATHS,
        batch_size=BATCH_SIZE,
        label_names=LABEL_COLUMNS,
        cat_names = CATEGORICAL_COLUMNS,
        cont_names=CONTINUOUS_COLUMNS,
        engine='parquet',
        shuffle=False,
        buffer_size=0.06,
        parts_per_chunk=1
    )

    inputs = {}    # tf.keras.Input placeholders for each feature to be used
    emb_layers = []# output of all embedding layers, which will be concatenated
    num_layers = []# output of numerical layers

    for col in CATEGORICAL_COLUMNS:
        inputs[col] =  tf.keras.Input(
            name=col,
            dtype=tf.int32,
            shape=(1,)
        )
    
    for col in CONTINUOUS_COLUMNS:
        inputs[col] =  tf.keras.Input(
            name=col,
            dtype=tf.float32,
            shape=(1,)
        )

    
    for col in CATEGORICAL_COLUMNS:
        emb_layers.append(
            tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_identity(
                    col, 
                    EMBEDDING_TABLE_SHAPES[col][0]                    # Input dimension (vocab size)
                ), EMBEDDING_TABLE_SHAPES[col][1]                     # Embedding output dimension
            )
        )

    for col in CONTINUOUS_COLUMNS:
        num_layers.append(
            tf.feature_column.numeric_column(col)
        )

    emb_layer = layers.DenseFeatures(emb_layers)
    x_emb_output = emb_layer(inputs)
    x_emb_output
 
    x = tf.keras.layers.Dense(128, activation="relu")(x_emb_output)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile('sgd', 'binary_crossentropy')

    tf.keras.utils.plot_model(model)
    
    validation_callback = KerasSequenceValidater(valid_dataset_tf)

    history = model.fit(train_dataset_tf, callbacks=[validation_callback], epochs=1)

    model.save(os.path.join(input_path, "model.savedmodel"))

def train_pytorch():
    from fastai.basics import Learner
    from fastai.tabular.model import TabularModel
    from fastai.tabular.data import TabularDataLoaders
    from fastai.metrics import RocAucBinary, APScoreBinary
    from fastai.callback.progress import ProgressCallback
    
    train_data_itrs = TorchAsyncItr(
        train_data,
        batch_size=BATCH_SIZE,
        cats=CATEGORICAL_COLUMNS,
        conts=CONTINUOUS_COLUMNS,
        labels=LABEL_COLUMNS,
        parts_per_chunk=PARTS_PER_CHUNK
    )
    valid_data_itrs = TorchAsyncItr(
        valid_data,
        batch_size=BATCH_SIZE,
        cats=CATEGORICAL_COLUMNS,
        conts=CONTINUOUS_COLUMNS,
        labels=LABEL_COLUMNS,
        parts_per_chunk=PARTS_PER_CHUNK
    )

    def gen_col(batch):
        return (batch[0], batch[1], batch[2].long())

    train_dataloader = DLDataLoader(train_data_itrs, collate_fn=gen_col, batch_size=None, pin_memory=False, num_workers=0)
    valid_dataloader = DLDataLoader(valid_data_itrs, collate_fn=gen_col, batch_size=None, pin_memory=False, num_workers=0)
    databunch = TabularDataLoaders(train_dataloader, valid_dataloader)

    embeddings = list(get_embedding_sizes(workflow).values())
    # We limit the output dimension to 16
    embeddings = [[emb[0], min(16, emb[1])] for emb in embeddings]
    embeddings

    model = TabularModel(emb_szs=embeddings, n_cont=len(CONTINUOUS_COLUMNS), out_sz=2, layers=[512, 256]).cuda()
    learn =  Learner(databunch, model, loss_func = torch.nn.CrossEntropyLoss(), metrics=[RocAucBinary(), APScoreBinary()])

    learning_rate = 1.32e-2
    epochs = 1
    start = time()
    learn.fit(epochs, learning_rate)
    t_final = time() - start
    total_rows = train_data_itrs.num_rows_processed + valid_data_itrs.num_rows_processed
    print(f"run_time: {t_final} - rows: {total_rows} - epochs: {epochs} - dl_thru: {total_rows / t_final}")

def main(args):
    # Set up data paths
    input_path = args.data_path[:-1] if args.data_path[-1] == "/" else args.data_path
    base_dir = args.out_path[:-1] if args.out_path[-1] == "/" else args.out_path
    dask_workdir = os.path.join(base_dir, "workdir")
    output_path = os.path.join(base_dir, "output")
    stats_path = os.path.join(base_dir, "stats")

    # Make sure we have a clean worker space for Dask
    if os.path.isdir(dask_workdir):
        shutil.rmtree(dask_workdir)
    os.makedirs(dask_workdir)

    # Make sure we have a clean stats space for Dask
    if os.path.isdir(stats_path):
        shutil.rmtree(stats_path)
    os.mkdir(stats_path)
            
    # Make sure we have a clean output path
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    # Get cats/conts/labels
    config = json.load(args.config_file)
    cats = config['cats']
    conts = config['conts']
    labels = config['label']

    # Use total device size to calculate args.device_limit_frac
    device_size = device_mem_size(kind="total")
    device_limit = int(args.device_limit_frac * device_size)
    device_pool_size = int(args.device_pool_frac * device_size)
    part_size = int(args.part_mem_frac * device_size)

    # Check if any device memory is already occupied
    for dev in args.devices.split(","):
        fmem = _pynvml_mem_size(kind="free", index=int(dev))
        used = (device_size - fmem) / 1e9
        if used > 1.0:
            warnings.warn(f"BEWARE - {used} GB is already occupied on device {int(dev)}!")

    # Setup LocalCUDACluster
    if args.protocol == "tcp":
        cluster = LocalCUDACluster(
            protocol=args.protocol,
            n_workers=len(args.devices.split(",")),
            CUDA_VISIBLE_DEVICES=args.devices,
            device_memory_limit=device_limit,
            local_directory=dask_workdir,
        )
    else:
        cluster = LocalCUDACluster(
            protocol=args.protocol,
            n_workers=len(args.devices.split(",")),
            CUDA_VISIBLE_DEVICES=args.devices,
            enable_nvlink=True,
            device_memory_limit=device_limit,
            local_directory=dask_workdir,
        )

    client = Client(cluster)

    # Setup RMM pool
    if args.device_pool_frac > 0.01:
        setup_rmm_pool(client, device_pool_size)

    # Define Dask NVTabular "Workflow"
    cont_features = cont_names >> ops.FillMissing() >> ops.Clip(min_value=0) >> ops.LogOp()

    cat_features = cat_names >> ops.Categorify(out_path=stats_path)

    workflow = Workflow(cat_features + cont_features + label_name, client=client)

    dataset = Dataset(input_path, "parquet", part_size=part_size)

    workflow.transform(dataset).to_parquet(output_path=output_path, shuffle=nvt_io.Shuffle.PER_WORKER, out_files_per_proc=args.out_files_per_proc)

    # Train with HugeCTR
    if args.target_framework == "hugectr":
        train_hugectr()
    # Train with TensorFLow
    elif args.target_framework == "tensorflow":
        train_tensorflow()
    # Train with PyTorch
    else:
        train_pytorch()

def parse_args():
    parser = argparse.ArgumentParser(description=("Dataset Test"))

    parser.add_argument("--data-path", type=str, help="Input dataset path (Required)")
    
    parser.add_argument("--out-path", type=str, help="Directory path to write output (Required)")

    parser.add_argument(
    "-d",
    "--devices",
    default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    type=str,
    help='Comma-separated list of visible devices (e.g. "0,1,2,3"). '
    "The number of visible devices dictates the number of Dask workers (GPU processes) "
    "The CUDA_VISIBLE_DEVICES environment variable will be used by default",
    )
    
    parser.add_argument("--target-framework", type=str, help="Target Framework (Required)")
    
    parser.add_argument("--config-file", type=str, help="Configuration file (Required)")

    parser.add_argument(
        "-p",
        "--protocol",
        choices=["tcp", "ucx"],
        default="tcp",
        type=str,
        help="Communication protocol to use (Default 'tcp')",
    )
    
    parser.add_argument(
        "--device-limit-frac",
        default=0.8,
        type=float,
        help="Worker device-memory limit as a fraction of GPU capacity (Default 0.8). "
        "The worker will try to spill data to host memory beyond this limit",
    )
    
    parser.add_argument(
        "--device-pool-frac",
        default=0.9,
        type=float,
        help="RMM pool size for each worker  as a fraction of GPU capacity (Default 0.9). "
        "If 0 is specified, the RMM pool will be disabled",
    )

    parser.add_argument(
        "--part-mem-frac",
        default=0.125,
        type=float,
        help="Maximum size desired for dataset partitions as a fraction "
        "of GPU capacity (Default 0.125)",
    )

    parser.add_argument(
        "--out-files-per-proc",
        default=8,
        type=int,
        help="Number of output files to write on each worker (Default 8)",
    )