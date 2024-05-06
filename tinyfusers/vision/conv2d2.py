import cudnn
import cupy as cp
import numpy as np
import math
from typing import Optional, Union, Tuple, cast
from tinygrad.tensor import Tensor
from tinygrad.helpers import prod

handle = cudnn.create_handle()

def conv_2d(X_gpu, W_gpu, padding, stride, dilation, bias):
    graph = cudnn.pygraph(handle = handle, name = "cudnn_graph_conv2d", 
                          io_data_type = cudnn.data_type.FLOAT, 
                          intermediate_data_type = cudnn.data_type.FLOAT, 
                          compute_data_type = cudnn.data_type.FLOAT)
    HW = W_gpu.shape[2:]
    #print(f"\n{HW}, {bias.shape}, {bias.reshape(1, -1, *[1] * len(HW))}")
    X_gpu = cp.asarray(X_gpu.numpy())
    W_gpu = cp.asarray(W_gpu.numpy())
    print(f"\nInput: {X_gpu.shape}\nWeight: {W_gpu.shape}\n")
    X = graph.tensor_like(X_gpu)
    W = graph.tensor_like(W_gpu)
    Y = graph.conv_fprop(image = X, weight = W, padding = padding, stride = stride, dilation = dilation, compute_data_type = cudnn.data_type.FLOAT)
    Y.set_output(True)
    graph.build([cudnn.heur_mode.A])
    Y_actual = cp.zeros(Y.get_dim(), dtype=cp.float32)
    workspace = cp.empty(graph.get_workspace_size(), dtype=cp.uint8)
    graph.execute({X: X_gpu, W: W_gpu, Y: Y_actual}, workspace, handle= handle)
    np_actual = cp.asnumpy(Y_actual)
    Y_tensor = Tensor(np_actual)
    return (Y_tensor, np_actual) if bias is None else (Y_tensor.add(bias.reshape(1, -1, *[1] * len(HW))), np_actual)

class Conv2d:
  def __init__(self, in_channels, out_channels, kernel_size, stride=[1,1], padding=[0,0], dilation=[1,1], groups=1, bias=True):
    self.kernel_size = kernel_size
    self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups
    self.weight = self.initialize_weight(out_channels, in_channels, groups)
    bound = 1 / math.sqrt(cast(int, prod(self.weight.shape[1:])))  # weight shape is always ints but mypy cannot tell
    self.bias = Tensor.uniform(out_channels, low=-bound, high=bound) if bias else None
    self.bias = None
  def __call__(self, x:Tensor):
      ddout, np_actual = conv_2d(x, self.weight, padding=self.padding, stride=self.stride, dilation=self.dilation, bias=self.bias)
      ttout = x.conv2d(self.weight, self.bias, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)
      print(f"\n{ddout}\n{ttout}\n")
      print(f"Compare: {np.allclose(ddout.numpy(), ttout.numpy(), atol=1e-2, rtol=1e-2)}")
      print(f"Compare: {np.allclose(np_actual, ttout.numpy(), atol=1e-1, rtol=1e-1)}")
      return ddout
  def initialize_weight(self, out_channels, in_channels, groups):
    return Tensor.kaiming_uniform(out_channels, in_channels//groups, *self.kernel_size, a=math.sqrt(5))
