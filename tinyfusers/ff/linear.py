import ctypes
import numpy as np
import cudnn
import cupy as cp
from ..native.cublas.ops import cublas
from ..native.cuda.ops import cudart
from ..storage.tensor import Tensor

def linear(X_gpu, W_gpu, B_gpu):  
    handle = cudnn.create_handle()  
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

def linear_cublas(weight, x):
  handle = cublas.cublasHandle_t()
  status = cublas.cublasCreate(handle)
  if status != 0:
    raise RuntimeError('cublasCreate_v2 failed with status {}'.format(status))
  
  t_op_a = cublas.CUBLAS_OP_T
  t_op_b = cublas.CUBLAS_OP_T
  alpha, beta = 1.0, 0.0

  m, k = weight.shape[0], weight.shape[1] # (m,k)
  _, n = x.shape[0], x.shape[1] # (k, n)
  
  lda, ldb, ldc = k, n, m
  res = Tensor.zeros((m*n), dtype = np.float32).eval() #(m,n)

  d_x = x.dt_ptr
  d_w = weight.dt_ptr
  d_res = res.dt_ptr

  status = cublas.cublasSgemm(handle, t_op_a, t_op_b,  m, n, k, alpha, 
                                d_w, lda, 
                                d_x, ldb, beta, 
                                d_res, ldc)
  if status != 0:
      raise RuntimeError('cublasSgemm_v2 failed with status {}'.format(status))

  cublas.cublasDestroy(handle)
  return res

class Linear:
  def __init__(self, in_features, out_features, bias=True):
    self.weight = cp.ones((out_features, in_features), dtype=cp.float32)
    self.bias = cp.ones((out_features), dtype=cp.float32) if bias else None
  def __call__(self, x):
    if x.device =="cuda":
       return linear_cublas(x.eval(), self.weight.eval()) + self.bias
    weight = cp.transpose(self.weight)
    o_tf = cp.dot(x, weight) + self.bias if self.bias is not None else cp.dot(x, weight)
    return o_tf