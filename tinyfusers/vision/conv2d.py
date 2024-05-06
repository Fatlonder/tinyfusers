import cudnn
import cupy as cp
import math
from tinygrad.tensor import Tensor

handle = cudnn.create_handle()

def conv_2d(X_gpu, W_gpu, padding, stride, dilation):
    graph = cudnn.pygraph(handle = handle, name = "cudnn_graph_conv2d", 
                          io_data_type = cudnn.data_type.HALF, 
                          intermediate_data_type = cudnn.data_type.HALF, 
                          compute_data_type = cudnn.data_type.FLOAT)
    X = graph.tensor_like(X_gpu)
    W = graph.tensor_like(W_gpu)
    Y = graph.conv_fprop(image = X, weight = W, padding = padding, stride = stride, dilation = dilation, compute_data_type = cudnn.data_type.FLOAT)
    Y.set_output(True)
    graph.build([cudnn.heur_mode.A])
    Y_actual = cp.zeros(Y.get_dim(), dtype=cp.float16)
    workspace = cp.empty(graph.get_workspace_size(), dtype=cp.uint8)
    graph.execute({X: X_gpu, W: W_gpu, Y: Y_actual}, workspace, handle=handle)
    return Y_actual

class Conv2d:
  def __init__(self, in_channels, out_channels, kernel_size, stride=[1,1], padding=[0,0], dilation=[1,1], groups=1, bias=False):
    self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups
    self.weight = self.initialize_weight(out_channels, in_channels, groups)
  def __call__(self, x):
    return conv_2d(x, self.weight, padding=self.padding, stride=self.stride, dilation=self.dilation)
  def initialize_weight(self, out_channels, in_channels, groups):
    return Tensor.kaiming_uniform(out_channels, in_channels//groups, *self.kernel_size, a=math.sqrt(5))
