import functools, operator
import cudnn
import cupy as cp
import math
from typing import cast

handle = cudnn.create_handle()

def conv_2d(X_gpu, W_gpu, padding, stride, dilation):
    graph = cudnn.pygraph(handle = handle, name = "conv2d", 
                          io_data_type = cudnn.data_type.FLOAT, 
                          intermediate_data_type = cudnn.data_type.FLOAT, 
                          compute_data_type = cudnn.data_type.FLOAT)
    N, K = X_gpu.shape[0], W_gpu.shape[0]
    X = graph.tensor_like(X_gpu)
    W = graph.tensor_like(W_gpu)
    Y = graph.conv_fprop(image = X, weight = W, padding = padding, stride = stride, dilation = dilation, compute_data_type = cudnn.data_type.FLOAT)
    Y.set_output(True)
    graph.build([cudnn.heur_mode.A])
    Y_actual = cp.zeros(Y.get_dim(), dtype=cp.float32)
    workspace = cp.empty(graph.get_workspace_size(), dtype=cp.uint8)
    graph.execute({X: X_gpu, W: W_gpu, Y: Y_actual}, workspace, handle= handle)
    o_cp = cp.transpose(cp.ravel(Y_actual).reshape(N, -1, K), (0, 2, 1)).reshape(Y_actual.shape)
    return o_cp


def conv_2d_16(X_gpu, W_gpu, padding, stride, dilation):
    graph = cudnn.pygraph(handle = handle, name = "conv2d_16", 
                          io_data_type = cudnn.data_type.HALF, 
                          intermediate_data_type = cudnn.data_type.HALF, 
                          compute_data_type = cudnn.data_type.FLOAT)
    N, K = X_gpu.shape[0], W_gpu.shape[0]
    X = graph.tensor_like(X_gpu)
    W = graph.tensor_like(W_gpu)
    Y = graph.conv_fprop(image = X, weight = W, padding = padding, stride = stride, dilation = dilation, compute_data_type = cudnn.data_type.FLOAT)
    Y.set_output(True)
    graph.build([cudnn.heur_mode.A])
    Y_actual = cp.zeros(Y.get_dim(), dtype=cp.float16)
    workspace = cp.empty(graph.get_workspace_size(), dtype=cp.uint8)
    graph.execute({X: X_gpu, W: W_gpu, Y: Y_actual}, workspace, handle= handle)
    o_cp = cp.transpose(cp.ravel(Y_actual).reshape(N, -1, K), (0, 2, 1)).reshape(Y_actual.shape)
    return o_cp

class Conv2d:
  def __init__(self, in_channels, out_channels, kernel_size, stride=[1,1], padding=[0,0], dilation=[1,1], groups=1, bias=True):
    self.kernel_size = kernel_size
    self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups
    self.weight =  cp.random.uniform(-math.sqrt(3.0), math.sqrt(3.0), (out_channels, in_channels // self.groups, *self.kernel_size), dtype=cp.float32)
    bound = 1 / math.sqrt(cast(int, functools.reduce(operator.mul, self.weight.shape[1:], 1)))
    self.bias = cp.random.uniform(-bound, bound, (out_channels,)) if bias else None
  def __call__(self, x):
      HW = self.weight.shape[2:]
      o_tf_cp = conv_2d(x, self.weight, padding=self.padding, stride=self.stride, dilation=self.dilation)
      o_tf = o_tf_cp if self.bias is None else o_tf_cp + (self.bias.reshape(1, -1, *[1] * len(HW)))
      return o_tf