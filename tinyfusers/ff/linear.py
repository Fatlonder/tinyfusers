import ctypes
import numpy as np
import cudnn
import cupy as cp
from ..native import cuda, cudart, nvrtc, cublas
from ..storage.tensor import Tensor

def gemm_batch(W, X):
    B, M, K = W.shape
    N = X.shape[2]
    status = cublas.cublasCreate(handle := cublas.cublasHandle_t())
    if status != 0:
      raise RuntimeError(f"cublasCreate_v2 failed with status {status}")

    A_gpu = (ctypes.POINTER(ctypes.c_float) * B)()
    B_gpu = (ctypes.POINTER(ctypes.c_float) * B)()
    C_gpu = (ctypes.POINTER(ctypes.c_float) * B)()

    for i in range(B):
      status_a = cudart.cudaMalloc(dd := ctypes.c_void_p(), M * K * np.dtype(np.float32).itemsize)
      A_gpu[i] = ctypes.cast(dd, ctypes.POINTER(ctypes.c_float))

      status_b = cudart.cudaMalloc(ff := ctypes.c_void_p(), K * N * np.dtype(np.float32).itemsize)
      B_gpu[i] = ctypes.cast(ff, ctypes.POINTER(ctypes.c_float))

      status_c = cudart.cudaMalloc(zz := ctypes.c_void_p(), M * N * np.dtype(np.float32).itemsize)
      C_gpu[i] = ctypes.cast(zz, ctypes.POINTER(ctypes.c_float))

      if (status_a or status_b or status_c) != 0:
        raise RuntimeError(f"cudaMalloc failed with status {status}")

    status_a = cudart.cudaMalloc(d_A_array := ctypes.c_void_p(), 2 * B * np.dtype(np.float32).itemsize)
    status_b = cudart.cudaMalloc(d_B_array := ctypes.c_void_p(), 2 * B * np.dtype(np.float32).itemsize)
    status_c = cudart.cudaMalloc(d_C_array := ctypes.c_void_p(), 2 * B * np.dtype(np.float32).itemsize)
    if (status_a or status_b or status_c) != 0:
        raise RuntimeError(f"cudaMalloc failed with status {status}")
    
    d_A_array = ctypes.cast(d_A_array, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))
    d_B_array = ctypes.cast(d_B_array, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))
    d_C_array = ctypes.cast(d_C_array, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))

    for i in range(B):
      status_a = cudart.cudaMemcpy(A_gpu[i], W.data[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)), M * K * np.dtype(np.float32).itemsize, cudart.cudaMemcpyHostToDevice)
      status_b = cudart.cudaMemcpy(B_gpu[i], X.data[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)), K * N * np.dtype(np.float32).itemsize, cudart.cudaMemcpyHostToDevice)
      if (status_a or status_b) != 0:
        raise RuntimeError(f"cudaMemcpy failed with status {status}")
      
    status_a = cudart.cudaMemcpy(d_A_array, A_gpu, 2 * B * np.dtype(np.float32).itemsize, cudart.cudaMemcpyHostToDevice)
    status_b = cudart.cudaMemcpy(d_B_array, B_gpu, 2 * B * np.dtype(np.float32).itemsize, cudart.cudaMemcpyHostToDevice)
    status_c = cudart.cudaMemcpy(d_C_array, C_gpu, 2 * B * np.dtype(np.float32).itemsize, cudart.cudaMemcpyHostToDevice)
    if (status_a or status_b or status_c) != 0:
        raise RuntimeError(f"cudaMemcpy failed with status {status}")

    t_op_a = cublas.CUBLAS_OP_T
    t_op_b = cublas.CUBLAS_OP_T
    alpha, beta = 1.0, 0.0
    lda, ldb, ldc = K, N, M
    status = cublas.cublasSgemmBatched(handle, t_op_a, t_op_b,  M, N, K, alpha, 
                                  d_A_array, lda, 
                                  d_B_array, ldb, beta, 
                                  d_C_array, ldc, B)
    if status != 0:
        raise RuntimeError(f"cublasSgemmBatched failed with status {status}")
    return C_gpu

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

def linear_cublas(weight, x, bias):
  status = cublas.cublasCreate(handle := cublas.cublasHandle_t())
  if status != 0:
    raise RuntimeError('cublasCreate_v2 failed with status {}'.format(status))
  
  t_op_a = cublas.CUBLAS_OP_T
  t_op_b = cublas.CUBLAS_OP_T
  alpha, beta = 1.0, 0.0

  m, k = weight.shape[0], weight.shape[1] # (m,k)
  _, n = x.shape[0], x.shape[1] # (k, n)
  
  lda, ldb, ldc = k, n, m
  res = Tensor.zeros((m*n), dtype = np.float32).eval() #(m,n). This incurs unecessary cost!
  d_bias = bias.dt_ptr if bias is not None else None

  status = cublas.cublasSgemm(handle, t_op_a, t_op_b,  m, n, k, alpha, 
                                weight.dt_ptr, lda, 
                                x.dt_ptr, ldb, beta, 
                                res.dt_ptr, ldc)
  if status != 0:
      raise RuntimeError(f"cublasSgemm_v2 failed with status {status}")
  
  if d_bias: # TODO Use fused op instead.
    bias.eval() 
    bias.device.add_bias(res.dt_ptr, bias.dt_ptr)
  
  cublas.cublasDestroy(handle)
  return res

class Linear:
  def __init__(self, in_features, out_features, bias=True):
    self.weight = cp.ones((out_features, in_features), dtype=cp.float32)
    self.bias = cp.ones((out_features), dtype=cp.float32) if bias else None
  def __call__(self, x):
    if x.device =="cuda":
       return linear_cublas(x.eval(), self.weight.eval(), self.bias)
    weight = cp.transpose(self.weight)
    o_tf = cp.dot(x, weight) + self.bias if self.bias is not None else cp.dot(x, weight)
    return o_tf