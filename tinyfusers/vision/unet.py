import functools
import cupy as cp
from .conv2d import Conv2d
from ..attention.attention import SpatialTransformer
from ..vision.resnet import ResBlock
from ..ff.linear import Linear
from ..ff.group_norm import GroupNorm
from ..tensor.tensor import Tensor

class UNetModel:
  def __init__(self):
    self.time_embed = [
      Linear(320, 1280),
      Tensor.silu,
      Linear(1280, 1280),
    ]
    self.input_blocks = [
      [Conv2d(4, 320, kernel_size=[3,3], padding=[1,1])],
      [ResBlock(320, 1280, 320), SpatialTransformer(320, 768, 8, 40)],
      [ResBlock(320, 1280, 320), SpatialTransformer(320, 768, 8, 40)],
      [Downsample(320)],
      [ResBlock(320, 1280, 640), SpatialTransformer(640, 768, 8, 80)],
      [ResBlock(640, 1280, 640), SpatialTransformer(640, 768, 8, 80)],
      [Downsample(640)],
      [ResBlock(640, 1280, 1280), SpatialTransformer(1280, 768, 8, 160)],
      [ResBlock(1280, 1280, 1280), SpatialTransformer(1280, 768, 8, 160)],
      [Downsample(1280)],
      [ResBlock(1280, 1280, 1280)],
      [ResBlock(1280, 1280, 1280)]
    ]
    self.middle_block = [
      ResBlock(1280, 1280, 1280),
      SpatialTransformer(1280, 768, 8, 160),
      ResBlock(1280, 1280, 1280)
    ]
    self.output_blocks = [
      [ResBlock(2560, 1280, 1280)],
      [ResBlock(2560, 1280, 1280)],
      [ResBlock(2560, 1280, 1280), Upsample(1280)],
      [ResBlock(2560, 1280, 1280), SpatialTransformer(1280, 768, 8, 160)],
      [ResBlock(2560, 1280, 1280), SpatialTransformer(1280, 768, 8, 160)],
      [ResBlock(1920, 1280, 1280), SpatialTransformer(1280, 768, 8, 160), Upsample(1280)],
      [ResBlock(1920, 1280, 640), SpatialTransformer(640, 768, 8, 80)],  # 6
      [ResBlock(1280, 1280, 640), SpatialTransformer(640, 768, 8, 80)],
      [ResBlock(960, 1280, 640), SpatialTransformer(640, 768, 8, 80), Upsample(640)],
      [ResBlock(960, 1280, 320), SpatialTransformer(320, 768, 8, 40)],
      [ResBlock(640, 1280, 320), SpatialTransformer(320, 768, 8, 40)],
      [ResBlock(640, 1280, 320), SpatialTransformer(320, 768, 8, 40)],
    ]
    self.out = [
      GroupNorm(32, 320),
      Tensor.silu,
      Conv2d(320, 4, kernel_size=[3,3], padding=[1,1])
    ]

  def __call__(self, x, timesteps=None, context=None):
    # TODO: real time embedding
    t_emb = timestep_embedding(timesteps, 320)
    emb = Tensor.sequential(self.time_embed, t_emb)
    #emb = t_emb.sequential(self.time_embed)

    def run(x, bb):
      if isinstance(bb, ResBlock): x = bb(x, emb)
      elif isinstance(bb, SpatialTransformer): x = bb(x, context)
      else: x = bb(x)
      return x

    saved_inputs = []
    for i,b in enumerate(self.input_blocks):
      #print("input block", i)
      for bb in b:
        x = run(x, bb)
      saved_inputs.append(x)
    for bb in self.middle_block:
      x = run(x, bb)
    for i,b in enumerate(self.output_blocks):
      #print("output block", i)
      x = cp.concatenate((x, saved_inputs.pop()), axis=1)
      for bb in b:
        x = run(x, bb)
    return Tensor.sequential(self.out, x)
  
class Upsample:
  def __init__(self, channels):
    self.conv = Conv2d(channels, channels, kernel_size=[3,3], padding=[1,1])
  def __call__(self, x):
    bs,c,py,px = x.shape
    x = cp.broadcast_to(x.reshape(bs, c, py, 1, px, 1), (bs, c, py, 2, px, 2)).reshape(bs, c, py*2, px*2)
    return self.conv(x)

class Downsample:
  def __init__(self, channels):
    self.op = Conv2d(channels, channels, stride=[2,2], kernel_size=[3,3], padding=[1,1])
  def __call__(self, x):
    return self.op(x)

def timestep_embedding(timesteps, dim, max_period=10000):
  half = dim // 2
  freqs = cp.exp(-cp.log(max_period) * cp.arange(half, dtype=cp.float32) / half)
  args = timesteps * freqs
  result = cp.concatenate((cp.cos(args), cp.sin(args))).reshape(1, -1)
  return result