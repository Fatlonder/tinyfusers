from ..vision.resnet import ResnetBlock
from ..attention.attention import AttnBlock
from ..tensor.tensor import Tensor

class Mid:
  def __init__(self, block_in):
    self.block_1 = ResnetBlock(block_in, block_in)
    self.attn_1 = AttnBlock(block_in)
    self.block_2 = ResnetBlock(block_in, block_in)

  def __call__(self, x):
    return Tensor.sequential([self.block_1, self.attn_1, self.block_2], x)