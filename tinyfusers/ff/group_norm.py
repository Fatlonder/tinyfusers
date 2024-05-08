from typing import Optional
from tinygrad.tensor import Tensor
import cupy as cp

def group_norm(x:Tensor, num_groups, eps):
    N, C, H, W = x.shape
    x_cupy = cp.asarray(x.numpy()).reshape(x_cupy, (N, num_groups, -1))
    mean = cp.mean(x_cupy, axis=-1, keepdims=True) #Calculate using Welford's Algorithm, on kernel model.  
    y_n = x_cupy-mean
    y_var = cp.sqrt(cp.mean(cp.square(y_n), axis=-1, keepdims=True) + eps)
    x = cp.reshape(y_n*(1/y_var), (N,C,H,W))
    return x
  

class GroupNorm:
  def __init__(self, num_groups:int, num_channels:int, eps:float=1e-5, affine:bool=True):
    self.num_groups, self.num_channels, self.eps = num_groups, num_channels, eps
    self.weight: Optional[Tensor] = Tensor.ones(num_channels) if affine else None
    self.bias: Optional[Tensor] = Tensor.zeros(num_channels) if affine else None
  def __call__(self, x:Tensor):
    return group_norm(x, self.num_groups, self.eps)*self.weight + self.bias