from tinygrad.nn import GroupNorm, Linear
from tinygrad import Tensor
from .conv2d4 import Conv2d


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
    h = x.sequential(self.in_layers)
    emb_out = emb.sequential(self.emb_layers)
    h = h + emb_out.reshape(*emb_out.shape, 1, 1)
    h = h.sequential(self.out_layers)
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
    h = self.conv1(self.norm1(x).swish())
    h = self.conv2(self.norm2(h).swish())
    return self.nin_shortcut(x) + h