import glob

import cupy

# Tensorflow
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# NVTabular
import nvtabular as nvt  # noqa: isort
from nvtabular.framework_utils.tensorflow import layers  # noqa: isort
from nvtabular.loader.tensorflow import KerasSequenceLoader  # noqa: isort

# Horovod
import horovod.tensorflow as hvd  # noqa: isort

hvd.init()

rank = hvd.rank()
local_rank = hvd.local_rank()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[local_rank], "GPU")

cupy.cuda.Device(local_rank).use()

BASE_DIR = "./datasets/"

BATCH_SIZE = 1024  # Batch Size
CATEGORICAL_COLUMNS = ["movieId", "userId"]  # Single-hot
CATEGORICAL_MH_COLUMNS = ["genres"]  # Multi-hot
NUMERIC_COLUMNS = []

TRAIN_PATHS = sorted(glob.glob(BASE_DIR + "train/*.parquet"))  # Output from ETL-with-NVTabular
VALID_PATHS = sorted(glob.glob(BASE_DIR + "valid/*.parquet"))  # Output from ETL-with-NVTabular

proc = nvt.Workflow.load(BASE_DIR + "workflow")

EMBEDDING_TABLE_SHAPES = nvt.ops.get_embedding_sizes(proc)

train_dataset_tf = KerasSequenceLoader(
    TRAIN_PATHS,  # you could also use a glob pattern
    batch_size=BATCH_SIZE,
    label_names=["rating"],
    cat_names=CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS,
    cont_names=NUMERIC_COLUMNS,
    engine="parquet",
    shuffle=True,
    buffer_size=0.06,  # how many batches to load at once
    devices=[local_rank],
    parts_per_chunk=1,
)

valid_dataset_tf = KerasSequenceLoader(
    VALID_PATHS,  # you could also use a glob pattern
    batch_size=BATCH_SIZE,
    label_names=["rating"],
    cat_names=CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS,
    cont_names=NUMERIC_COLUMNS,
    engine="parquet",
    shuffle=False,
    buffer_size=0.06,
    devices=[local_rank],
    parts_per_chunk=1,
)

inputs = {}  # tf.keras.Input placeholders for each feature to be used
emb_layers = []  # output of all embedding layers, which will be concatenated

for col in CATEGORICAL_COLUMNS:
    inputs[col] = tf.keras.Input(name=col, dtype=tf.int32, shape=(1,))
# Note that we need two input tensors for multi-hot categorical features
for col in CATEGORICAL_MH_COLUMNS:
    inputs[col + "__values"] = tf.keras.Input(name=f"{col}__values", dtype=tf.int64, shape=(1,))
    inputs[col + "__nnzs"] = tf.keras.Input(name=f"{col}__nnzs", dtype=tf.int64, shape=(1,))

for col in CATEGORICAL_COLUMNS:  # +CATEGORICAL_MH_COLUMNS:
    emb_layers.append(
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(
                col, EMBEDDING_TABLE_SHAPES[col][0]  # Input dimension (vocab size)
            ),
            # EMBEDDING_TABLE_SHAPES[col][1]                     # Embedding output dimension
            4,
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
    tape = hvd.DistributedGradientTape(tape)

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

    if batch % 10 == 0 and local_rank == 0:
        print("Step #%d\tLoss: %.6f" % (batch, loss_value))

# Horovod: save checkpoints only on worker 0 to prevent other workers from
# corrupting it.
if local_rank == 0:
    checkpoint.save(checkpoint_dir)
