import cudnn
import cupy as cp
import numpy as np
import math
from tinygrad.tensor import Tensor

handle = cudnn.create_handle()

def linear(X_gpu, W_gpu, B_gpu):    
    graph = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT,handle=handle,)
    M, K = X_gpu.shape[1], W_gpu.shape[2]
    X = graph.tensor(name="X", dim=X_gpu.size(), stride=X_gpu.stride(), data_type=X_gpu.dtype,)
    W = graph.tensor(name="W", dim=W_gpu.size(), stride=W_gpu.stride(), data_type=W_gpu.dtype,)
    B = graph.tensor(name="B", dim=B_gpu.size(), stride=B_gpu.stride(), data_type=B_gpu.dtype,)
    response = graph.matmul(name="matmul", A=X, B=W)
    Y = graph.bias(name="bias", input=response, bias=B)
    Y.set_output(True).set_data_type(cudnn.data_type.HALF)
    graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    workspace = cp.empty(graph.get_workspace_size(), dtype=cp.uint8)
    Y_actual = cp.empty((M, K), dtype=cp.float16)
    graph.execute({X: X_gpu, W: W_gpu, B: B_gpu, Y: Y_actual}, workspace, handle=handle)
    return Y_actual

class Linear:
  def __init__(self, in_features, out_features, bias=True):
    bound = 1 / math.sqrt(in_features)
    self.weight = Tensor.kaiming_uniform(out_features, in_features, a=math.sqrt(5))
    self.bias = Tensor.uniform(out_features, low=-bound, high=bound) if bias else Tensor(0)
  def __call__(self, x:Tensor):
    o_tg = x.linear(self.weight.transpose(), self.bias)
    cur_stream = cp.cuda.get_current_stream()
    cur_stream.use()
    x_cp = cp.asarray(x.numpy())
    weight = cp.asarray(self.weight.transpose().numpy())
    bias = cp.asarray(self.bias.numpy())
    cur_stream.synchronize()
    cp.cuda.Device().synchronize()
    o_tf = cp.dot(x_cp, weight) + bias 
    o_np = cp.asnumpy(o_tf)
    cur_stream.synchronize()
    cp.cuda.Device().synchronize()
    np.testing.assert_allclose(o_np, o_tg.numpy(), atol=1e-2, rtol=1e-2)
    o_t = Tensor(o_np).realize()
    return o_t