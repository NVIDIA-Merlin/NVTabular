import glob
import os
import typing as T
from dataclasses import dataclass

import dask_cudf


@dataclass
class TabularDataset(object):
    train_path: str
    eval_path: str

    categorical_features: T.List[str]
    continuous_features: T.List[str]
    targets: T.List[str]

    @property
    def train_files(self):
        return sorted(glob.glob(os.path.join(self.train_path, "*.parquet")))

    def train_df(self, sample=0.1):
        return dask_cudf.read_parquet(self.train_files).sample(frac=sample).compute()

    def train_tf_dataset(self, batch_size, separate_labels=True, named_labels=False, shuffle=True, buffer_size=0.06,
                         parts_per_chunk=1):
        from nvtabular.loader.tensorflow import KerasSequenceLoader

        output = KerasSequenceLoader(
            self.train_files,
            batch_size=batch_size,
            label_names=self.targets if separate_labels else [],
            cat_names=self.categorical_features if separate_labels else self.categorical_features + self.targets,
            cont_names=self.continuous_features,
            engine="parquet",
            shuffle=shuffle,
            buffer_size=buffer_size,  # how many batches to load at once
            parts_per_chunk=parts_per_chunk,
        )

        if named_labels and separate_labels:
            return output.map(lambda X, y: (X, dict(zip(self.targets, y))))

        return output

    @property
    def eval_files(self):
        return sorted(glob.glob(os.path.join(self.eval_path, "*.parquet")))

    def eval_df(self, sample=0.1):
        return dask_cudf.read_parquet(self.eval_files).sample(frac=sample).compute()

    def eval_tf_dataset(self, batch_size, separate_labels=True, named_labels=False, shuffle=True, buffer_size=0.06,
                        parts_per_chunk=1):
        from nvtabular.loader.tensorflow import KerasSequenceLoader

        output = KerasSequenceLoader(
            self.eval_files,
            batch_size=batch_size,
            label_names=self.targets if separate_labels else [],
            cat_names=self.categorical_features if separate_labels else self.categorical_features + self.targets,
            cont_names=self.continuous_features,
            engine="parquet",
            shuffle=shuffle,
            buffer_size=buffer_size,  # how many batches to load at once
            parts_per_chunk=parts_per_chunk,
        )

        if named_labels and separate_labels:
            return output.map(lambda X, y: (X, dict(zip(self.targets, y))))

        return output

    def eval_tf_callback(self, batch_size, **kwargs):
        from nvtabular.loader.tensorflow import KerasSequenceValidater

        return KerasSequenceValidater(self.eval_tf_dataset(batch_size, **kwargs))

    # def create_default_input_layer(self, workflow: nvt.Workflow):
    #     return InputLayer(self.continuous_features, EmbeddingsLayer.from_nvt_workflow(workflow))

    def create_keras_inputs(self, for_prediction=False, sparse_columns=None):
        import tensorflow as tf

        if sparse_columns is None:
            sparse_columns = []
        inputs = {}

        for col in self.continuous_features:
            inputs[col] = tf.keras.Input(name=col, dtype=tf.float32, shape=(None, 1))

        for col in self.categorical_features:
            if for_prediction or col not in sparse_columns:
                inputs[col] = tf.keras.Input(name=col, dtype=tf.int32, shape=(None, 1))
            else:
                inputs[col + "__values"] = tf.keras.Input(name=f"{col}__values", dtype=tf.int64, shape=(1,))
                inputs[col + "__nnzs"] = tf.keras.Input(name=f"{col}__nnzs", dtype=tf.int64, shape=(1,))

        return inputs
