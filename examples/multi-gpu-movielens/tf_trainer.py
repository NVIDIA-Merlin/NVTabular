# External dependencies
import argparse
import glob
import os

import cupy

# we can control how much memory to give tensorflow with this environment variable
# IMPORTANT: make sure you do this before you initialize TF's runtime, otherwise
# TF will have claimed all free GPU memory
os.environ["TF_MEMORY_ALLOCATION"] = "0.3"  # fraction of free memory

import nvtabular as nvt  # noqa: E402 isort:skip
from nvtabular.framework_utils.tensorflow import layers  # noqa: E402 isort:skip
from nvtabular.loader.tensorflow import KerasSequenceLoader  # noqa: E402 isort:skip

import tensorflow as tf  # noqa: E402 isort:skip
import horovod.tensorflow as hvd  # noqa: E402 isort:skip

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--dir_in", default=None, help="Input directory")
parser.add_argument("--batch_size", default=None, help="batch size")
parser.add_argument("--cats", default=None, help="categorical columns")
parser.add_argument("--cats_mh", default=None, help="categorical multihot columns")
parser.add_argument("--conts", default=None, help="continuous columns")
parser.add_argument("--labels", default=None, help="continuous columns")
args = parser.parse_args()


BASE_DIR = args.dir_in or "./data/"
BATCH_SIZE = int(args.batch_size or 16384)  # Batch Size
CATEGORICAL_COLUMNS = args.cats or ["movieId", "userId"]  # Single-hot
CATEGORICAL_MH_COLUMNS = args.cats_mh or ["genres"]  # Multi-hot
NUMERIC_COLUMNS = args.conts or []
TRAIN_PATHS = sorted(
    glob.glob(os.path.join(BASE_DIR, "train/*.parquet"))
)  # Output from ETL-with-NVTabular
hvd.init()

# Seed with system randomness (or a static seed)
cupy.random.seed(None)


def seed_fn():
    """
    Generate consistent dataloader shuffle seeds across workers

    Reseeds each worker's dataloader each epoch to get fresh a shuffle
    that's consistent across workers.
    """
    min_int, max_int = tf.int32.limits
    max_rand = max_int // hvd.size()

    # Generate a seed fragment on each worker
    seed_fragment = cupy.random.randint(0, max_rand).get()

    # Aggregate seed fragments from all Horovod workers
    seed_tensor = tf.constant(seed_fragment)
    reduced_seed = hvd.allreduce(seed_tensor, name="shuffle_seed", op=hvd.mpi_ops.Sum)

    return reduced_seed % max_rand


proc = nvt.Workflow.load(os.path.join(BASE_DIR, "workflow/"))
EMBEDDING_TABLE_SHAPES, MH_EMBEDDING_TABLE_SHAPES = nvt.ops.get_embedding_sizes(proc)
EMBEDDING_TABLE_SHAPES.update(MH_EMBEDDING_TABLE_SHAPES)


train_dataset_tf = KerasSequenceLoader(
    TRAIN_PATHS,  # you could also use a glob pattern
    batch_size=BATCH_SIZE,
    label_names=["rating"],
    cat_names=CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS,
    cont_names=NUMERIC_COLUMNS,
    engine="parquet",
    shuffle=True,
    buffer_size=0.06,  # how many batches to load at once
    parts_per_chunk=1,
    global_size=hvd.size(),
    global_rank=hvd.rank(),
    seed_fn=seed_fn,
)
inputs = {}  # tf.keras.Input placeholders for each feature to be used
emb_layers = []  # output of all embedding layers, which will be concatenated
for col in CATEGORICAL_COLUMNS:
    inputs[col] = tf.keras.Input(name=col, dtype=tf.int32, shape=(1,))
# Note that we need two input tensors for multi-hot categorical features
for col in CATEGORICAL_MH_COLUMNS:
    inputs[col] = (
        tf.keras.Input(name=f"{col}__values", dtype=tf.int64, shape=(1,)),
        tf.keras.Input(name=f"{col}__nnzs", dtype=tf.int64, shape=(1,)),
    )
for col in CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS:
    emb_layers.append(
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(
                col, EMBEDDING_TABLE_SHAPES[col][0]
            ),  # Input dimension (vocab size)
            EMBEDDING_TABLE_SHAPES[col][1],  # Embedding output dimension
        )
    )
emb_layer = layers.DenseFeatures(emb_layers)
x_emb_output = emb_layer(inputs)
x = tf.keras.layers.Dense(128, activation="relu")(x_emb_output)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs=inputs, outputs=x)
loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.SGD(0.01 * hvd.size())
opt = hvd.DistributedOptimizer(opt)
checkpoint_dir = "./checkpoints"
checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)


@tf.function(experimental_relax_shapes=True)
def training_step(examples, labels, first_batch):
    with tf.GradientTape() as tape:
        probs = model(examples, training=True)
        loss_value = loss(labels, probs)
    # Horovod: add Horovod Distributed GradientTape.
    tape = hvd.DistributedGradientTape(tape, sparse_as_dense=True)
    grads = tape.gradient(loss_value, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    #
    # Note: broadcast should be done after the first gradient step to ensure optimizer
    # initialization.
    if first_batch:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)
    return loss_value


# Horovod: adjust number of steps based on number of GPUs.
for batch, (examples, labels) in enumerate(train_dataset_tf):
    loss_value = training_step(examples, labels, batch == 0)
    if batch % 100 == 0 and hvd.local_rank() == 0:
        print("Step #%d\tLoss: %.6f" % (batch, loss_value))
hvd.join()

# Horovod: save checkpoints only on worker 0 to prevent other workers from
# corrupting it.
if hvd.rank() == 0:
    checkpoint.save(checkpoint_dir)
