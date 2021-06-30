from typing import Optional, Callable, Text, Any, Dict

import torch

from nvtabular.framework_utils.torch import TabularModule, FilterFeatures


class TableConfig(object):
    def __init__(self,
                 vocabulary_size: int,
                 dim: int,
                 # initializer: Optional[Callable[[Any], None]],
                 # optimizer: Optional[_Optimizer] = None,
                 combiner: Text = "mean",
                 name: Optional[Text] = None):
        if not isinstance(vocabulary_size, int) or vocabulary_size < 1:
            raise ValueError("Invalid vocabulary_size {}.".format(vocabulary_size))

        if not isinstance(dim, int) or dim < 1:
            raise ValueError("Invalid dim {}.".format(dim))

        if combiner not in ("mean", "sum", "sqrtn"):
            raise ValueError("Invalid combiner {}".format(combiner))

        self.vocabulary_size = vocabulary_size
        self.dim = dim
        self.combiner = combiner
        self.name = name

    def __repr__(self):
        return (
            "TableConfig(vocabulary_size={vocabulary_size!r}, dim={dim!r}, "
            "combiner={combiner!r}, name={name!r})".format(
                vocabulary_size=self.vocabulary_size,
                dim=self.dim,
                combiner=self.combiner,
                name=self.name, )
        )


class FeatureConfig(object):
    def __init__(self,
                 table: TableConfig,
                 max_sequence_length: int = 0,
                 name: Optional[Text] = None):
        if not isinstance(table, TableConfig):
            raise ValueError("table is type {}, expected "
                             "`tf.tpu.experimental.embedding.TableConfig`".format(
                type(table)))

        if not isinstance(max_sequence_length, int) or max_sequence_length < 0:
            raise ValueError("Invalid max_sequence_length {}.".format(
                max_sequence_length))

        self.table = table
        self.max_sequence_length = max_sequence_length
        self.name = name

    def __repr__(self):
        return (
            "FeatureConfig(table={table!r}, "
            "max_sequence_length={max_sequence_length!r}, name={name!r})"
                .format(
                table=self.table,
                max_sequence_length=self.max_sequence_length,
                name=self.name)
        )


class EmbeddingsModule(TabularModule):
    def __init__(self, feature_config: Dict[str, FeatureConfig], **kwargs):
        super().__init__(**kwargs)
        self.feature_config = feature_config
        self.filter_features = FilterFeatures(list(feature_config.keys()))

        embedding_tables = {}
        tables: Dict[str, TableConfig] = {}
        for name, feature in self.feature_config.items():
            table: TableConfig = feature.table
            if table.name not in tables:
                tables[table.name] = table

        for name, table in tables.items():
            embedding_tables[name] = torch.nn.EmbeddingBag(table.vocabulary_size, table.dim, mode=table.combiner)

        self.embedding_tables = torch.nn.ModuleDict(embedding_tables)

    def forward(self, inputs):
        embedded_outputs = {}
        filtered_inputs = self.filter_features(inputs)
        for name, val in filtered_inputs.items():
            if isinstance(val, tuple):
                values, offsets = val
                values = torch.squeeze(values, -1)
                # for the case where only one value in values
                if len(values.shape) == 0:
                    values = values.unsqueeze(0)
                embedded_outputs[name] = self.embedding_tables[name](values, offsets[:, 0])
            else:
                if len(val.shape) <= 1:
                    val = val.unsqueeze(0)
                embedded_outputs[name] = self.embedding_tables[name](val)

        return embedded_outputs
