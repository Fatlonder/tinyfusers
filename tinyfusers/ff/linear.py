import math
import ctypes
import numpy as np
import cudnn
import cupy as cp
from ..native import cuda, cudart, nvrtc, cublas
from ..storage.tensor import Tensor

add_bias_cuda_src = '''
extern "C"
__global__ 
void add_bias(float* out, const float* bias, int BT, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BT * OC){
      int col = idx % OC;
      out[((idx % OC) * BT) + idx/OC] += bias[col];
    }
}'''

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
  res = Tensor.zeros((m*n), dtype = np.float32).eval() #(m,n). This incurs unecessary cost!

  d_x = x.dt_ptr
  d_w = weight.dt_ptr
  d_res = res.dt_ptr
  d_bias = bias.dt_ptr if bias is not None else None

  status = cublas.cublasSgemm(handle, t_op_a, t_op_b,  m, n, k, alpha, 
                                d_w, lda, 
                                d_x, ldb, beta, 
                                d_res, ldc)
  if status != 0:
      raise RuntimeError('cublasSgemm_v2 failed with status {}'.format(status))
  
  if d_bias: # TODO Use fused op instead.
    bias.eval() 
    device_id = 0 

    status = cuda.cuInit(handler := cuda.handler())
    if status != 0:
      raise RuntimeError(f"cuInit failed with status {status}")
    status = cuda.cuCtxCreate_v2(cntx := cuda.CUcontext(), cuda.CU_CTX_SCHED_AUTO, device_id)
    if status != 0:
      raise RuntimeError(f"cuCtxCreate_v2 failed with status {status}")
    
    cudart.cudaDeviceGetAttribute(ctypes.byref(major := ctypes.c_int32()), 75, device_id)
    cudart.cudaDeviceGetAttribute(ctypes.byref(minor := ctypes.c_int32()), 76, device_id)

    use_cubin = (minor.value >= 1)
    prefix = 'sm' if use_cubin else 'compute'
    compile_options = [ bytes(f'--gpu-architecture={prefix}_{major.value}{minor.value}', 'ascii'),
                      bytes('-I/usr/local/cuda/include', 'ascii'),
                      bytes('-I/usr/include', 'ascii'), bytes('-I/opt/cuda/include/', 'ascii')]

    status = nvrtc.nvrtcCreateProgram(prog := nvrtc.nvrtcProgram(), add_bias_cuda_src)
    if status != 0:
      raise RuntimeError(f"nvrtcCreateProgram failed with status {status}")
    
  status = nvrtc.nvrtcCompileProgram(prog, compile_options)
  if status != 0:
    raise RuntimeError(f"nvrtcCompileProgram failed with status {status}")
  
  if use_cubin:
    nvrtc.nvrtcGetCUBINSize(prog, (dataSize := ctypes.c_ulong()))
    data = b' ' * dataSize.value
    status = nvrtc.nvrtcGetCUBIN(prog, data)
    if status != 0:
      raise RuntimeError(f"nvrtcGetCUBIN failed with status {status}")
  else:
      dataSize = nvrtc.nvrtcGetPTXSize(prog)
      data = b' ' * dataSize
      status = nvrtc.nvrtcGetPTX(prog, data)
      if status != 0:
        raise RuntimeError(f"nvrtcGetPTX failed with status {status}")
      
  status = cuda.cuModuleLoadData(module := cuda.CUmodule(), data)
  if status != 0:
    raise RuntimeError(f"cuModuleLoadData failed with status {status}")

  status  = cuda.cuModuleGetFunction(add_bias_fnc := cuda.CUfunction(), module, b"add_bias")
  if status != 0:
    raise RuntimeError(f"cuModuleGetFunction failed with status {status}")

  block_x, block_y, block_z = 5, 1, 1
  grid_x, grid_y, grid_z = math.ceil(m*n/ float(block_x)), 1, 1
  kernelArgs = [ctypes.addressof(d_res), ctypes.addressof(bias.dt_ptr), 
                ctypes.cast(ctypes.pointer(ctypes.c_uint32(m)), ctypes.c_void_p), 
                ctypes.cast(ctypes.pointer(ctypes.c_uint32(n)), ctypes.c_void_p)]
  c_array = (ctypes.c_void_p * len(kernelArgs))(*kernelArgs)
  status = cuda.cuLaunchKernel(add_bias_fnc, grid_x, grid_y, grid_z,    # grid dim
                                      block_x, block_y, block_z, # block dim
                                      0, stream := cuda.CUstream(),                 # shared mem and stream
                                      c_array, None)
  if status != 0:
    raise RuntimeError(f"cuLaunchKernel failed with status {status}")
  
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