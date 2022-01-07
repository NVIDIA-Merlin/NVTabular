from distutils.spawn import find_executable

import pytest

torch = pytest.importorskip("torch")  # noqa
configure_tensorflow = pytest.importorskip("nvtabular.loader.tf_utils.configure_tensorflow")  # noqa
configure_tensorflow()

import nvtabular.framework_utils.tensorflow.layers as layers  # noqa
from nvtabular.framework_utils.torch.models import Model  # noqa

triton = pytest.importorskip("nvtabular.inference.triton")
data_conversions = pytest.importorskip("nvtabular.inference.triton.data_conversions")
ensemble = pytest.importorskip("nvtabular.inference.triton.ensemble")

tritonclient = pytest.importorskip("tritonclient")
grpcclient = pytest.importorskip("tritonclient.grpc")

TRITON_SERVER_PATH = find_executable("tritonserver")
from tests.unit.test_triton_inference import run_triton_server  # noqa

tf = pytest.importorskip("tensorflow")


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


def _run_ensemble_on_tritonserver(
    tmpdir,
    output_columns,
    df,
    model_name,
):
    inputs = triton.convert_df_to_triton_input(df.columns, df)
    outputs = [grpcclient.InferRequestedOutput(col) for col in output_columns]
    response = None
    with run_triton_server(tmpdir) as client:
        response = client.infer(model_name, inputs, outputs=outputs)

    return response
