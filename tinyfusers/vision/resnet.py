import cupy as cp
from .conv2d import Conv2d
from ..ff.linear import Linear
from ..ff.group_norm import GroupNorm
from ..tensor.tensor import Tensor

class ResBlock:
  def __init__(self, channels, emb_channels, out_channels):
    self.in_layers = [
      GroupNorm(32, channels),
      Tensor.silu,
      Conv2d(channels, out_channels, kernel_size=[3,3], padding=[1,1])
    ]
    self.emb_layers = [
      Tensor.silu,
      Linear(emb_channels, out_channels)
    ]
    self.out_layers = [
      GroupNorm(32, out_channels),
      Tensor.silu,
      lambda x: x,  # needed for weights loading code to work
      Conv2d(out_channels, out_channels, kernel_size=[3,3], padding=[1,1])
    ]
    self.skip_connection = Conv2d(channels, out_channels, kernel_size=[1,1]) if channels != out_channels else lambda x: x

  def __call__(self, x, emb):
    h = Tensor.sequential(self.in_layers, x)
    emb_out = Tensor.sequential(self.emb_layers, emb)
    h = h + emb_out.reshape(*emb_out.shape, 1, 1).astype(cp.float32)
    h = Tensor.sequential(self.out_layers, h)
    ret = self.skip_connection(x) + h
    return ret

class ResnetBlock:
  def __init__(self, in_channels, out_channels=None):
    self.norm1 = GroupNorm(32, in_channels)
    self.conv1 = Conv2d(in_channels, out_channels, kernel_size=[3,3], padding=[1,1])
    self.norm2 = GroupNorm(32, out_channels)
    self.conv2 = Conv2d(out_channels, out_channels, kernel_size=[3,3], padding=[1,1])
    self.nin_shortcut = Conv2d(in_channels, out_channels, kernel_size=[1,1]) if in_channels != out_channels else lambda x: x

  def __call__(self, x):
    h = self.conv1(Tensor.swish(self.norm1(x)))
    h = self.conv2(Tensor.swish(self.norm2(h)))
    return self.nin_shortcut(x) + h