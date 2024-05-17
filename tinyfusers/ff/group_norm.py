from typing import Optional
from tinygrad.tensor import Tensor
import cupy as cp
import numpy as np

def group_norm(x:Tensor, num_groups, eps):
    N, C, H, W = x.shape
    x_cupy = cp.reshape(cp.asarray(x.numpy()), (N, num_groups, -1))
    cp.cuda.runtime.deviceSynchronize()
    mean = cp.mean(x_cupy, axis=-1, keepdims=True) #Calculate using Welford's Algorithm, on kernel model.  
    y_n = x_cupy-mean
    y_var = cp.sqrt(cp.mean(cp.square(y_n), axis=-1, keepdims=True) + eps)
    x = cp.reshape(y_n*(1/y_var), (N,C,H,W))
    Y_tensor = Tensor(cp.asnumpy(x))
    return Y_tensor
  

class GroupNorm:
  def __init__(self, num_groups:int, num_channels:int, eps:float=1e-5, affine:bool=True):
    self.num_groups, self.num_channels, self.eps = num_groups, num_channels, eps
    self.weight: Optional[Tensor] = Tensor.ones(num_channels) if affine else None
    self.bias: Optional[Tensor] = Tensor.zeros(num_channels) if affine else None
  def __call__(self, x:Tensor):
    o_tf = group_norm(x, self.num_groups, self.eps)
    x = x.reshape(x.shape[0], self.num_groups, -1).layernorm(eps=self.eps).reshape(x.shape)
    if self.weight is None or self.bias is None:
      o_tg = x
    else:
      sc = self.weight.reshape(1, -1, *[1] * (len(x.shape)-2)) + self.bias.reshape(1, -1, *[1] * (len(x.shape)-2))
      o_tg = x * sc
      o_tf = o_tf * sc
    np.testing.assert_allclose(o_tg.numpy(), o_tf.numpy(), atol=1e-2, rtol=1e-2)
    return o_tg