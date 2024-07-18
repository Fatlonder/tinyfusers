import math
import numpy as np
import ctypes
from tinyfusers.native import cuda, cudart, nvrtc

np.random.seed(42)
B, T, C, OC = 1, 10, 10, 1*10

add_vec_cuda = '''\
extern "C"
__global__ void vectorAddGPU(const float *a, const float *b, float *c, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < N) {
        c[idx] =  a[idx] + b[idx];
    }
}'''

inp_bytes = C*np.dtype(np.float32).itemsize
weight_bytes = C*np.dtype(np.float32).itemsize
out_bytes = C*np.dtype(np.float32).itemsize

inp = np.ones(C).astype(np.float32)
inp = np.array([1,1,1,1,1,2,2,2,2,2]).astype(np.float32)
weight = np.ones(C).astype(np.float32)
out = np.zeros(C).astype(np.float32)

device_id = 0
cntx = cuda.CUcontext()
handler = cuda.handler()
module  = cuda.CUmodule()
stream = cuda.CUstream()
add_vector = cuda.CUfunction()

status = cuda.cuInit(handler)
if status != 0:
  raise RuntimeError('cuInit failed with status {}'.format(status))

status = cuda.cuCtxCreate_v2(cntx, cuda.CU_CTX_SCHED_AUTO, device_id)
if status != 0:
  raise RuntimeError('cuCtxCreate_v2 failed with status {}'.format(status))

cudart.cudaDeviceGetAttribute(ctypes.byref(major := ctypes.c_int32()), 75, device_id)
cudart.cudaDeviceGetAttribute(ctypes.byref(minor := ctypes.c_int32()), 76, device_id)

use_cubin = (minor.value >= 1)
prefix = 'sm' if use_cubin else 'compute'
compile_options = [ bytes(f'--gpu-architecture={prefix}_{major.value}{minor.value}', 'ascii'),
                   bytes('-I/usr/local/cuda/include', 'ascii'),
                   bytes('-I/usr/include', 'ascii'), bytes('-I/opt/cuda/include/', 'ascii')]

status = nvrtc.nvrtcCreateProgram(prog := nvrtc.nvrtcProgram(), add_vec_cuda)
if status != 0:
  raise RuntimeError('nvrtcCreateProgram failed with status {}'.format(status))

status = nvrtc.nvrtcCompileProgram(prog, compile_options)
if status != 0:
  raise RuntimeError('nvrtcCompileProgram failed with status {}'.format(status))

if use_cubin:
    nvrtc.nvrtcGetCUBINSize(prog, (dataSize := ctypes.c_ulong()))
    data = b' ' * dataSize.value
    status = nvrtc.nvrtcGetCUBIN(prog, data)
    if status != 0:
      raise RuntimeError('nvrtcGetCUBIN failed with status {}'.format(status))
else:
    dataSize = nvrtc.nvrtcGetPTXSize(prog)
    data = b' ' * dataSize
    status = nvrtc.nvrtcGetPTX(prog, data)
    if status != 0:
      raise RuntimeError('nvrtcGetPTX failed with status {}'.format(status))

status = cuda.cuModuleLoadData(module, data)
if status != 0:
  raise RuntimeError('cuModuleLoadData failed with status {}'.format(status))

status  = cuda.cuModuleGetFunction(add_vector, module, b"vectorAddGPU")
if status != 0:
  raise RuntimeError('cuModuleGetFunction failed with status {}'.format(status))


cudart.cudaMalloc((d_a := ctypes.c_void_p()), inp_bytes)
cudart.cudaMalloc((d_b := ctypes.c_void_p()), weight_bytes)
cudart.cudaMalloc((d_c := ctypes.c_void_p()), out_bytes)

cudart.cudaMemcpy(d_a, inp.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p)), inp_bytes, cudart.cudaMemcpyHostToDevice)
cudart.cudaMemcpy(d_b, weight.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p)), weight_bytes, cudart.cudaMemcpyHostToDevice)

block_x, block_y, block_z = 8, 8, 1
grid_x, grid_y, grid_z = math.ceil(B * T / float(block_x)), math.ceil(OC / float(block_y)), 1

kernelArgs = [ctypes.addressof(d_a), ctypes.addressof(d_b), ctypes.addressof(d_c), ctypes.cast(ctypes.pointer(ctypes.c_uint32(OC)), ctypes.c_void_p)]
c_array = (ctypes.c_void_p * len(kernelArgs))(*kernelArgs)

status = cuda.cuLaunchKernel(add_vector, grid_x, grid_y, grid_z,    # grid dim
                                    block_x, block_y, block_z, # block dim
                                    0, stream,                 # shared mem and stream
                                    c_array, None)
if status != 0:
  raise RuntimeError('cuLaunchKernel failed with status {}'.format(status))

cudart.cudaMemcpy(out.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p)), d_c, weight_bytes, cudart.cudaMemcpyDeviceToHost)

print(out)