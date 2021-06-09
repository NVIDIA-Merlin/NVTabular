import pdb
import os

import cudf
import rmm
import tensorflow as tf
from merlin_models.tensorflow.models.retrieval import YouTubeDNN

# we can control how much memory to give tensorflow with this environment variable
# IMPORTANT: make sure you do this before you initialize TF's runtime, otherwise
# TF will have claimed all free GPU memory
os.environ["TF_MEMORY_ALLOCATION"] = "0.7"  # fraction of free memory

import nvtabular as nvt  # noqa: isort
from nvtabular.loader.tensorflow import KerasSequenceLoader, KerasSequenceValidater  # noqa: isort

rmm.reinitialize(managed_memory=True)

BASE_DIR = os.environ.get("BASE_DIR", os.path.expanduser("~/nvt-examples/end-to-end-poc/"))

INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", os.path.expanduser(f"{BASE_DIR}/data/"))

examples = cudf.read_parquet(os.path.join(INPUT_DATA_DIR, "grouped_examples.parquet"))

BATCH_SIZE = 2048  # Batch Size
CATEGORICAL_COLUMNS = []  # Single-hot
CATEGORICAL_MH_COLUMNS = ["sampled_tag", "movieId", "genre"]  # Multi-hot
NUMERIC_COLUMNS = ["movieId_count"]

movie_workflow = nvt.Workflow.load(os.path.join(INPUT_DATA_DIR, "movie_features_workflow"))

EMBEDDING_TABLE_SHAPES, MH_EMBEDDING_TABLE_SHAPES = nvt.ops.get_embedding_sizes(movie_workflow)
EMBEDDING_TABLE_SHAPES.update(MH_EMBEDDING_TABLE_SHAPES)

EMBEDDING_TABLE_SHAPES["sampled_tag"] = EMBEDDING_TABLE_SHAPES.pop("tags_unique", None)
EMBEDDING_TABLE_SHAPES["genre"] = EMBEDDING_TABLE_SHAPES.pop("genres", None)

train_dataset_tf = KerasSequenceLoader(
    os.path.join(INPUT_DATA_DIR, "grouped_examples.parquet"),
    batch_size=BATCH_SIZE,
    label_names=["target_item"],
    cat_names=CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS,
    cont_names=NUMERIC_COLUMNS,
    engine="parquet",
    shuffle=True,
    buffer_size=0.25,
    parts_per_chunk=1,
)

continuous_cols = []

for col in NUMERIC_COLUMNS:
    continuous_cols.append(tf.feature_column.numeric_column(col))

embedding_dims = {}

for key, value in EMBEDDING_TABLE_SHAPES.items():
    embedding_dims[key] = 128  # value[1]  # Latent dimensions

categorical_cols = []

for col in CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS:
    categorical_cols.append(
        tf.feature_column.categorical_column_with_identity(
            col, EMBEDDING_TABLE_SHAPES[col][0]  # Cardinalities
        )
    )

model = YouTubeDNN(
    continuous_cols, categorical_cols, embedding_dims=embedding_dims, hidden_dims=[512, 256, 128]
)

model.input_layer.build({})
item_embeddings = model.input_layer.embedding_tables["movieId"]


def sampled_softmax_loss(y_true, y_pred):
    return tf.nn.sampled_softmax_loss(
        weights=item_embeddings,
        biases=tf.zeros((item_embeddings.shape[0],)),
        labels=y_true,
        inputs=y_pred,
        num_sampled=100,
        num_classes=item_embeddings.shape[0],
    )


model.compile("adam", sampled_softmax_loss)

history = model.fit(train_dataset_tf, callbacks=[], epochs=100)

MODEL_BASE_DIR = os.environ.get("MODEL_BASE_DIR", os.path.expanduser(f"{BASE_DIR}/models/"))

MODEL_NAME_TF = os.environ.get("MODEL_NAME_TF", "movielens_youtube_retrieval")
MODEL_PATH_TEMP_TF = os.path.join(MODEL_BASE_DIR, MODEL_NAME_TF, "1/model.savedmodel")

model.save(MODEL_PATH_TEMP_TF)


# TODO: Construct a FAISS index
# TODO: Generate a Triton config?

rmm.reinitialize(managed_memory=False)
