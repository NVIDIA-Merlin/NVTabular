import tensorflow as tf


from nvtabular.framework_utils.tensorflow.heads import Head
from nvtabular.framework_utils.tensorflow.blocks.base import BlockType
from nvtabular.framework_utils.tensorflow.tfrs import Model


class BlockWithHead(Model):
    def __init__(self, block: BlockType, head: Head, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block = block
        self.head = head

    def call(self, inputs, **kwargs):
        return self.head(self.block(inputs, **kwargs), **kwargs)

    def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
        targets = self.head.pop_labels(inputs)
        logits = self(inputs, training=training)

        return self.head.compute_loss(targets, logits)

    def get_config(self):
        pass