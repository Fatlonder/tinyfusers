import cupy as cp

def group_norm(x, num_groups, eps):
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
    self.weight = cp.ones(num_channels, dtype=cp.float32) if affine else None
    self.bias = cp.zeros(num_channels, dtype=cp.float32) if affine else None
  def __call__(self, x):
    o_cp = group_norm(x, self.num_groups, self.eps)
    o_cp  = o_cp * self.weight.reshape(1, -1, *[1] * (len(x.shape)-2)) + self.bias.reshape(1, -1, *[1] * (len(x.shape)-2))
    return o_cp