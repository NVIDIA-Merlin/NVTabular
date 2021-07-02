from nvtabular.column_group import ColumnGroup

from merlin_models.tf.features.continuous import ContinuousFeatures
from merlin_models.tf.features.embedding import EmbeddingFeatures
from merlin_models.tf.features.text import TextEmbeddingFeaturesWithTransformers
from merlin_models.tf.tabular import TabularLayer
from nvtabular.tag import Tag


class TabularFeatures(TabularLayer):
    def __init__(self, continuous_layer=None, categorical_layer=None, text_embedding_layer=None, aggregation=None,
                 **kwargs):
        super(TabularFeatures, self).__init__(aggregation=aggregation, **kwargs)
        self.categorical_layer = categorical_layer
        self.continuous_layer = continuous_layer
        self.text_embedding_layer = text_embedding_layer

        self.to_apply = []
        if continuous_layer:
            self.to_apply.append(continuous_layer)
        if categorical_layer:
            self.to_apply.append(categorical_layer)
        if text_embedding_layer:
            self.to_apply.append(text_embedding_layer)

        assert (self.to_apply is not []), "Please provide at least one input layer"

    def call(self, inputs, **kwargs):
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
        maybe_continuous_layer, maybe_categorical_layer = None, None
        if continuous_tags:
            maybe_continuous_layer = ContinuousFeatures.from_column_group(
                column_group,
                tags=continuous_tags,
                tags_to_filter=continuous_tags_to_filter)
        if categorical_tags:
            maybe_categorical_layer = EmbeddingFeatures.from_column_group(
                column_group,
                tags=categorical_tags,
                tags_to_filter=categorical_tags_to_filter)

        if text_model and not isinstance(text_model, TextEmbeddingFeaturesWithTransformers):
            text_model = TextEmbeddingFeaturesWithTransformers.from_column_group(
                column_group,
                tags=text_tags,
                tags_to_filter=text_tags_to_filter,
                transformer_model=text_model,
                max_text_length=max_text_length)

        return cls(continuous_layer=maybe_continuous_layer,
                   categorical_layer=maybe_categorical_layer,
                   text_embedding_layer=text_model,
                   aggregation=aggregation,
                   **kwargs)

    def compute_output_shape(self, input_shapes):
        output_shapes = {}
        for in_layer in self.to_apply:
            output_shapes.update(in_layer.compute_output_shape(input_shapes))

        return super().compute_output_shape(output_shapes)