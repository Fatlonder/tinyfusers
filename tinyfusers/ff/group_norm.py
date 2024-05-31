from typing import Optional
from tinygrad.tensor import Tensor
import cupy as cp
import numpy as np
from tinygrad.nn import GroupNorm as GNorm

def group_norm(x:Tensor, num_groups, eps):
    N, C, H, W = x.shape
    x_cp = cp.reshape(x, (N, num_groups, -1))
    cp.cuda.runtime.deviceSynchronize()
    mean = cp.mean(x_cp, axis=-1, keepdims=True) #Calculate using Welford's Algorithm, on kernel model.  
    y_n = x_cp-mean
    y_var = cp.sqrt(cp.mean(cp.square(y_n), axis=-1, keepdims=True) + eps)
    o_cp = cp.reshape(y_n*(1/y_var), (N,C,H,W))
    return o_cp
  
class GroupNorm:
  def __init__(self, num_groups:int, num_channels:int, eps:float=1e-5, affine:bool=True):
    self.num_groups, self.num_channels, self.eps = num_groups, num_channels, eps
    self.weight: Optional[Tensor] = Tensor.ones(num_channels) if affine else None
    self.bias: Optional[Tensor] = Tensor.zeros(num_channels) if affine else None
  def __call__(self, x:Tensor):
    tg_gnorm = GNorm(self.num_groups, self.num_channels)
    tg_gnorm.weight = self.weight
    tg_gnorm.bias = self.bias
    o_tg = tg_gnorm(x)
    cur_stream = cp.cuda.get_current_stream()
    cur_stream.use()
    x_cp = cp.asarray(x.numpy())
    cur_stream.synchronize()
    cp.cuda.Device().synchronize()
    o_cp = group_norm(x_cp, self.num_groups, self.eps)
    o_np = cp.asnumpy(o_cp)
    cur_stream.synchronize()
    cp.cuda.Device().synchronize()
    o_tf = Tensor(o_np)
    o_tf = o_tf * self.weight.reshape(1, -1, *[1] * (len(x.shape)-2)) + self.bias.reshape(1, -1, *[1] * (len(x.shape)-2))
    np.testing.assert_allclose(o_tg.numpy(), o_tf.numpy(), atol=1e-2, rtol=1e-2)
    return o_tg