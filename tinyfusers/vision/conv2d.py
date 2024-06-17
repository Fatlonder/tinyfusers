import cudnn
import cupy as cp
import numpy as np
import math
from typing import cast
from tinygrad.tensor import Tensor
from tinygrad.helpers import prod

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
    self.weight = self.initialize_weight(out_channels, in_channels, groups)
    bound = 1 / math.sqrt(cast(int, prod(self.weight.shape[1:])))  # weight shape is always ints but mypy cannot tell
    self.bias = Tensor.uniform(out_channels, low=-bound, high=bound) if bias else None
  def __call__(self, x:Tensor):
      cur_stream = cp.cuda.get_current_stream()
      cur_stream.use()
      HW = self.weight.shape[2:]
      x_cp = cp.asarray(x.numpy()).astype(cp.float32)
      w_cp = cp.asarray(self.weight.numpy()).astype(cp.float32)
      cur_stream.synchronize()
      cp.cuda.Device().synchronize()
      o_tf_cp = conv_2d(x_cp, w_cp, padding=self.padding, stride=self.stride, dilation=self.dilation)
      o_tg = x.conv2d(self.weight, self.bias, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)
      o_np = cp.asnumpy(o_tf_cp)
      cur_stream.synchronize()
      cp.cuda.Device().synchronize()
      o_tf = Tensor(o_np)
      o_tf = o_tf if self.bias is None else o_tf.add(self.bias.reshape(1, -1, *[1] * len(HW))).realize()
      cur_stream.synchronize()
      cp.cuda.Device().synchronize()
      np.testing.assert_allclose(o_tg.numpy(), o_tf.numpy(), atol=1e-2, rtol=1e-2)
      return o_tf
  def initialize_weight(self, out_channels, in_channels, groups):
    return Tensor.kaiming_uniform(out_channels, in_channels//groups, *self.kernel_size, a=math.sqrt(5))