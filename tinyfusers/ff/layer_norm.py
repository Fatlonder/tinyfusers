import cudnn
import cupy as cp
import numpy as np
from typing import Union, Tuple
from tinygrad.tensor import Tensor

handle = cudnn.create_handle()

def layer_norm(x_gpu, scale_gpu, bias_gpu, epsilon_cpu):
    x_gpu_shape, scale_gpu_shape, bias_gpu_shape = x_gpu.shape, scale_gpu.shape, bias_gpu.shape
    x_gpu_stride = tuple([x_gpu.shape[1]*x_gpu.shape[2]*x_gpu.shape[3], 1, x_gpu.shape[1]*x_gpu.shape[3], x_gpu.shape[1]])
    scale_gpu_stride = tuple([scale_gpu.shape[1]*scale_gpu.shape[2]*scale_gpu.shape[3], 1, scale_gpu.shape[1]*scale_gpu.shape[3], scale_gpu.shape[1]])
    bias_gpu_stride = tuple([bias_gpu.shape[1]*bias_gpu.shape[2]*bias_gpu.shape[3], 1, bias_gpu.shape[1]*bias_gpu.shape[3], bias_gpu.shape[1]])
    epsilon_cpu_stride = tuple([1,1,1,1])
        
    x_gpu = cp.asarray(x_gpu.numpy())
    scale_gpu = cp.asarray(scale_gpu.numpy())
    bias_gpu = cp.asarray(bias_gpu.numpy())
    
    graph = cudnn.pygraph(intermediate_data_type = cudnn.data_type.FLOAT, compute_data_type = cudnn.data_type.FLOAT)
    X = graph.tensor(name = "X", dim = x_gpu_shape, stride = x_gpu_stride, data_type = cudnn.data_type.FLOAT)
    scale = graph.tensor(name = "scale", dim = scale_gpu_shape, stride = scale_gpu_stride, data_type = cudnn.data_type.FLOAT)
    bias = graph.tensor(name = "bias", dim = bias_gpu_shape, stride = bias_gpu_stride, data_type = cudnn.data_type.FLOAT)
    epsilon = graph.tensor(name = "epsilon", dim = epsilon_cpu.shape, stride = epsilon_cpu_stride, is_pass_by_value = True, data_type = cudnn.data_type.FLOAT)
    
    Y, mean, inv_var = graph.layernorm(name = "layer_norm", 
                        norm_forward_phase = cudnn.norm_forward_phase.INFERENCE,
                        input = X,
                        scale = scale, 
                        bias = bias,
                        epsilon = epsilon)
    Y.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    Y_actual = cp.empty(X.get_dim(), dtype=cp.float32)
    workspace = cp.empty(graph.get_workspace_size(), dtype=cp.uint8)
    graph.execute({X : x_gpu, scale : scale_gpu, bias : bias_gpu , epsilon: epsilon_cpu , Y : Y_actual}, workspace, handle=handle)
    return cp.asnumpy(Y_actual)

class LayerNorm:
  def __init__(self, normalized_shape:Union[int, Tuple[int, ...]], eps:float=1e-5, elementwise_affine:bool=True):
    self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    self.axis, self.eps, self.elementwise_affine = tuple(-1-i for i in range(len(self.normalized_shape))), eps, elementwise_affine
    self.weight, self.bias = (Tensor.ones(*self.normalized_shape), Tensor.zeros(*self.normalized_shape)) if elementwise_affine else (None, None)

  def __call__(self, x:Tensor):
    assert self.normalized_shape == x.shape[-len(self.normalized_shape):], f"last dimensions of {x.shape} must match {self.normalized_shape}"
    out_shape = x.shape
    x = x.layernorm(eps=self.eps.item(), axis=self.axis)
    x = x if not self.elementwise_affine else x * self.weight + self.bias
    y_tf = layer_norm(x.unsqueeze(1), self.weight.unsqueeze(0).unsqueeze(0).unsqueeze(0), self.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), self.eps.numpy())
    #np.testing.assert_allclose(x.numpy(), y_tf.reshape(x.shape), atol=1e-2, rtol=1e-2)
    ttt = Tensor(y_tf.reshape(out_shape), device="cuda")
    return ttt
