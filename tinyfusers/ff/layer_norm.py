import cudnn
import cupy as cp
from typing import Union, Tuple
from tinygrad.tensor import Tensor

handle = cudnn.create_handle()

def layer_norm(x_gpu, scale_gpu, bias_gpu, epsilon_cpu):
    graph = cudnn.pygraph(intermediate_data_type = cudnn.data_type.FLOAT, compute_data_type = cudnn.data_type.FLOAT)
    X = graph.tensor(name = "X", dim = x_gpu.size(), stride = x_gpu.stride(), data_type = x_gpu.dtype)
    scale = graph.tensor(name = "scale", dim = scale_gpu.size(), stride = scale_gpu.stride(), data_type = scale_gpu.dtype)
    bias = graph.tensor(name = "bias", dim = bias_gpu.size(), stride = bias_gpu.stride(), data_type = bias_gpu.dtype)
    epsilon = graph.tensor(name = "epsilon", dim = epsilon_cpu.size(), stride = epsilon_cpu.stride(), is_pass_by_value = True, data_type = epsilon_cpu.dtype)
    Y, mean, inv_var = graph.layernorm(name = "layer_norm", 
                        norm_forward_phase = cudnn.norm_forward_phase.INFERENCE,
                        input = X,
                        scale = scale, 
                        bias = bias,
                        epsilon = epsilon)
    Y.set_output(True).set_data_type(x_gpu.dtype)
    graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    Y_actual = cp.empty(X.get_dim(), dtype=cp.float32)
    workspace = cp.empty(graph.get_workspace_size(), dtype=cp.uint8)
    graph.execute({X : x_gpu, scale : scale_gpu, bias : bias_gpu , epsilon: epsilon_cpu , Y : Y_actual}, workspace, handle=handle)
    return Y_actual


class LayerNorm:
  def __init__(self, normalized_shape:Union[int, Tuple[int, ...]], eps:float=1e-5, elementwise_affine:bool=True):
    self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    self.axis, self.eps, self.elementwise_affine = tuple(-1-i for i in range(len(self.normalized_shape))), eps, elementwise_affine
    self.weight, self.bias = (Tensor.ones(*self.normalized_shape), Tensor.zeros(*self.normalized_shape)) if elementwise_affine else (None, None)

  def __call__(self, x:Tensor):
    assert self.normalized_shape == x.shape[-len(self.normalized_shape):], f"last dimensions of {x.shape} must match {self.normalized_shape}"
    x = x.layernorm(eps=self.eps, axis=self.axis)
    x = x if not self.elementwise_affine else x * self.weight + self.bias
    #xt = layer_norm(x, self.weight, self.bias, self.eps)
    return x
