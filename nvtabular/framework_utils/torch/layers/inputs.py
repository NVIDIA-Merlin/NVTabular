from typing import Optional, Callable, Text, Any, Dict

import torch

from nvtabular.column_group import ColumnGroup
from nvtabular.framework_utils.torch.features import TabularModule, FilterFeatures
from nvtabular.tag import Tag


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

    @classmethod
    def from_column_group(cls, column_group: ColumnGroup, embedding_dims=None, default_embedding_dim=64,
                          infer_embedding_sizes=True, combiner="mean", tags=None, tags_to_filter=None,
                          **kwargs) -> Optional["EmbeddingsModule"]:
        if tags:
            column_group = column_group.get_tagged(tags, tags_to_filter=tags_to_filter)

        if infer_embedding_sizes:
            sizes = column_group.embedding_sizes()
        else:
            if not embedding_dims:
                embedding_dims = {}
            sizes = {}
            cardinalities = column_group.cardinalities()
            for key, cardinality in cardinalities.items():
                embedding_size = embedding_dims.get(key, default_embedding_dim)
                sizes[key] = (cardinality, embedding_size)

        feature_config: Dict[str, FeatureConfig] = {}
        for name, (vocab_size, dim) in sizes.items():
            feature_config[name] = FeatureConfig(
                TableConfig(
                    vocabulary_size=vocab_size,
                    dim=dim,
                    name=name,
                    combiner=combiner,
                )
            )

        if not feature_config:
            return None

        return cls(feature_config, **kwargs)

    def forward(self, inputs, **kwargs):
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

    def forward_output_size(self, input_sizes):
        sizes = {}
        batch_size = self.calculate_batch_size_from_input_size(input_sizes)
        for name, feature in self.feature_config.items():
            sizes[name] = torch.Size([batch_size, feature.table.dim])

        return super().forward_output_size(sizes)


class InputFeatures(TabularModule):
    def __init__(self, continuous_module=None, categorical_module=None, text_embedding_module=None, aggregation=None):
        super(InputFeatures, self).__init__(aggregation=aggregation)
        self.categorical_module = categorical_module
        self.continuous_module = continuous_module
        self.text_embedding_module = text_embedding_module

        self.to_apply = []
        if continuous_module:
            self.to_apply.append(continuous_module)
        if categorical_module:
            self.to_apply.append(categorical_module)
        if text_embedding_module:
            self.to_apply.append(text_embedding_module)

        assert (self.to_apply is not []), "Please provide at least one input layer"

    def forward(self, inputs, **kwargs):
        return self.to_apply[0](inputs, merge_with=self.to_apply[1:] if len(self.to_apply) > 1 else None)

    @classmethod
    def from_column_group(cls,
                          column_group: ColumnGroup,
                          continuous_tags=Tag.CONTINUOUS,
                          continuous_tags_to_filter=None,
                          categorical_tags=Tag.CATEGORICAL,
                          categorical_tags_to_filter=None,
                          text_model=None,
                          text_tags=Tag.TEXT_TOKENIZED,
                          text_tags_to_filter=None,
                          max_text_length=None,
                          aggregation=None,
                          **kwargs):
        maybe_continuous_module, maybe_categorical_module = None, None
        if continuous_tags:
            maybe_continuous_module = TabularModule.from_column_group(
                column_group,
                tags=continuous_tags,
                tags_to_filter=continuous_tags_to_filter)
        if categorical_tags:
            maybe_categorical_module = EmbeddingsModule.from_column_group(
                column_group,
                tags=categorical_tags,
                tags_to_filter=categorical_tags_to_filter)

        # if text_model and not isinstance(text_model, TransformersTextEmbedding):
        #     text_model = TransformersTextEmbedding.from_column_group(
        #         column_group,
        #         tags=text_tags,
        #         tags_to_filter=text_tags_to_filter,
        #         transformer_model=text_model,
        #         max_text_length=max_text_length)

        return cls(continuous_module=maybe_continuous_module,
                   categorical_module=maybe_categorical_module,
                   text_embedding_module=text_model,
                   aggregation=aggregation)

    def forward_output_size(self, input_size):
        output_sizes = {}
        for in_layer in self.to_apply:
            output_sizes.update(in_layer.forward_output_size(input_size))

        return super().forward_output_size(output_sizes)
