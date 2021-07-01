from typing import Union

from nvtabular.framework_utils.torch.blocks.base import BuildableBlock, TabularBlock, Block


class MLPBlock(BuildableBlock):

    def __init__(self, dims) -> None:
        super().__init__()
        self.dims = dims

    def build(self, input_shape) -> Union[TabularBlock, Block]:
        print("build!")
        return TabularBlock()
