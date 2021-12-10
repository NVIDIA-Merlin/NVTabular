from distutils.spawn import find_executable

import pytest

import nvtabular as nvt
import nvtabular.framework_utils.tensorflow.layers as layers
import nvtabular.ops as ops
from nvtabular.framework_utils.torch.models import Model
from nvtabular.loader.tf_utils import configure_tensorflow

triton = pytest.importorskip("nvtabular.inference.triton")
data_conversions = pytest.importorskip("nvtabular.inference.triton.data_conversions")
ensemble = pytest.importorskip("nvtabular.inference.triton.ensemble")

from nvtabular.inference.triton.ensemble import (  # noqa
    export_pytorch_ensemble,
    export_tensorflow_ensemble,
)
from tests.unit.inference.test_ensemble import _run_ensemble_on_tritonserver  # noqa

tritonclient = pytest.importorskip("tritonclient")
grpcclient = pytest.importorskip("tritonclient.grpc")

TRITON_SERVER_PATH = find_executable("tritonserver")
configure_tensorflow()
tf = pytest.importorskip("tensorflow")


@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("output_model", ["tensorflow"])
def test_export_run_ensemble_triton(tmpdir, engine, output_model, df):
    conts = ["x", "y", "id"] >> ops.FillMissing() >> ops.Normalize()
    cats = ["name-cat", "name-string"] >> ops.Categorify(cat_cache="host")
    workflow = nvt.Workflow(conts + cats)
    nvt_dataset = nvt.Dataset(df)
    workflow.fit(nvt_dataset)

    embed_shapes = nvt.ops.get_embedding_sizes(workflow)
    cat_cols = list(embed_shapes.keys())

    if output_model == "tensorflow":
        tf_model = create_tf_model(cat_cols, [], embed_shapes)
        export_tensorflow_ensemble(tf_model, workflow, "test_name", tmpdir, [])
    elif output_model == "pytorch":
        torch_model = create_pytorch_model(cat_cols, [], embed_shapes)
        export_pytorch_ensemble(
            torch_model,
            workflow,
            {},
            "test_name",
            tmpdir,
            [],
        )

    # assert os.path.exists(os.path.join(repo, "config.pbtxt"))
    tri_df = df.iloc[:10]
    tri_df = tri_df[["x", "y", "id", "name-cat", "name-string"]]
    response = _run_ensemble_on_tritonserver(str(tmpdir), ["output"], tri_df, "test_name")
    assert response is not None
    assert len(response.as_numpy("output")) == 10


def create_tf_model(cat_columns: list, cat_mh_columns: list, embed_tbl_shapes: dict):
    inputs = {}  # tf.keras.Input placeholders for each feature to be used
    emb_layers = []  # output of all embedding layers, which will be concatenated
    for col in cat_columns:
        inputs[col] = tf.keras.Input(name=col, dtype=tf.int32, shape=(1,))
    # Note that we need two input tensors for multi-hot categorical features
    for col in cat_mh_columns:
        inputs[col] = (
            tf.keras.Input(name=f"{col}__values", dtype=tf.int64, shape=(1,)),
            tf.keras.Input(name=f"{col}__nnzs", dtype=tf.int64, shape=(1,)),
        )
    for col in cat_columns + cat_mh_columns:
        emb_layers.append(
            tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_identity(
                    col, embed_tbl_shapes[col][0]
                ),  # Input dimension (vocab size)
                embed_tbl_shapes[col][1],  # Embedding output dimension
            )
        )
    emb_layer = layers.DenseFeatures(emb_layers)
    x_emb_output = emb_layer(inputs)
    x = tf.keras.layers.Dense(128, activation="relu")(x_emb_output)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile("sgd", "binary_crossentropy")
    return model


def create_pytorch_model(cat_columns: list, cat_mh_columns: list, embed_tbl_shapes: dict):
    single_hot = {k: v for k, v in embed_tbl_shapes.items() if k in cat_columns}
    multi_hot = {k: v for k, v in embed_tbl_shapes.items() if k in cat_mh_columns}
    model = Model(
        embedding_table_shapes=(single_hot, multi_hot),
        num_continuous=0,
        emb_dropout=0.0,
        layer_hidden_dims=[128, 128, 128],
        layer_dropout_rates=[0.0, 0.0, 0.0],
    ).to("cuda")
    return model
