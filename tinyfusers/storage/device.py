import math
import ctypes, ctypes.util
import numpy as np
import os
import sys
from ..native import cuda, cudart, nvrtc, cublas

'''
Will be used to interface with device specific runtimes.
'''
class Device:
    def __init__(self, device:str):
        '''
        CUDA_HOME = os.getenv('CUDA_HOME')
        if CUDA_HOME == None:
            CUDA_HOME = os.getenv('CUDA_PATH')
        if CUDA_HOME == None:
            raise RuntimeError('Environment variable CUDA_HOME or CUDA_PATH is not set')
        '''
        self.device = device
        self.func_lib = {}
        self.cuda_kernel_dir = os.path.abspath('tinyfusers/tinyfusers/native/cuda/')
        self.device_id = 0
    
    def load_library_from_path(self, device, lib_path, lib_name = "cublas"):
        return 0

    def __str__(self):
        return str(self.device)
    
    def load_func(self, code_str, func_name):
        status = cuda.cuInit(handler := cuda.handler())
        if status != 0:
            raise RuntimeError(f"cuInit failed with status {status}")
        status = cuda.cuCtxCreate_v2(cntx := cuda.CUcontext(), cuda.CU_CTX_SCHED_AUTO, self.device_id)
        if status != 0:
            raise RuntimeError(f"cuCtxCreate_v2 failed with status {status}")
        
        cudart.cudaDeviceGetAttribute(ctypes.byref(major := ctypes.c_int32()), 75, self.device_id)
        cudart.cudaDeviceGetAttribute(ctypes.byref(minor := ctypes.c_int32()), 76, self.device_id)

        use_cubin = (minor.value >= 1)
        prefix = 'sm' if use_cubin else 'compute'
        compile_options = [ bytes(f'--gpu-architecture={prefix}_{major.value}{minor.value}', 'ascii'),
                        bytes('-I/usr/local/cuda/include', 'ascii'),
                        bytes('-I/usr/include', 'ascii'), bytes('-I/opt/cuda/include/', 'ascii')]

        status = nvrtc.nvrtcCreateProgram(prog := nvrtc.nvrtcProgram(), code_str)
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

        status  = cuda.cuModuleGetFunction(fnc_pointer := cuda.CUfunction(), module, bytes(func_name, 'ascii'))
        if status != 0:
            raise RuntimeError(f"cuModuleGetFunction failed with status {status}")
        
        return fnc_pointer

    def add_bias(self, result_pntr, bias_pntr, m, n):
        if "add_bias" in self.func_lib:
            add_bias_fnc = self.func_lib["add_bias"]
        else:
            cuda_kernel_file = os.path.join(self.cuda_kernel_dir, 'add_bias_func.cu')
            def read_file_content(code_filename):
                with open(code_filename, 'r') as f: return f.read()
            add_bias_fnc = self.load_func(read_file_content(cuda_kernel_file), "add_bias")
            self.func_lib["add_bias"] = add_bias_fnc

        block_x, block_y, block_z = 5, 1, 1
        grid_x, grid_y, grid_z = math.ceil(m*n/ float(block_x)), 1, 1
        kernelArgs = [ctypes.addressof(result_pntr), ctypes.addressof(bias_pntr), 
                    ctypes.cast(ctypes.pointer(ctypes.c_uint32(m)), ctypes.c_void_p), 
                    ctypes.cast(ctypes.pointer(ctypes.c_uint32(n)), ctypes.c_void_p)]
        c_array = (ctypes.c_void_p * len(kernelArgs))(*kernelArgs)

        status = cuda.cuLaunchKernel(add_bias_fnc, grid_x, grid_y, grid_z,    # grid dim
                                            block_x, block_y, block_z, # block dim
                                            0, stream := cuda.CUstream(),                 # shared mem and stream
                                            c_array, None)
        if status != 0:
            raise RuntimeError(f"cuLaunchKernel failed with status {status}")
        return result_pntr
    
    def scale_tensor(self, x_ptr, scaler, B, T, NH, OC):
        if "scale_kernel" in self.func_lib:
            scale_tensor_fnc = self.func_lib["scale_kernel"]
        else:
            cuda_kernel_file = os.path.join(self.cuda_kernel_dir, 'scale_tensor_func.cu')
            def read_file_content(code_filename):
                with open(code_filename, 'r') as f: return f.read()
            scale_tensor_fnc = self.load_func(read_file_content(cuda_kernel_file), "scale_kernel")
            self.func_lib["scale_kernel"] = scale_tensor_fnc
        
        block_x, block_y, block_z = 8, 1, 1
        grid_x, grid_y, grid_z = math.ceil(B * T * NH * OC/ float(block_x)), 1, 1
        kernelArgs = [ctypes.addressof(x_ptr), ctypes.cast(ctypes.pointer(ctypes.c_float(scaler)), ctypes.c_void_p),
              ctypes.cast(ctypes.pointer(ctypes.c_uint32(B * T * NH * OC)), ctypes.c_void_p)]
        c_args = (ctypes.c_void_p * len(kernelArgs))(*kernelArgs)
        status = cuda.cuLaunchKernel(scale_tensor_fnc, grid_x, grid_y, grid_z,    # grid dim
                                            block_x, block_y, block_z,            # block dim
                                            0, stream := cuda.CUstream(),         # shared mem and stream
                                            c_args, None)
        if status != 0:
            raise RuntimeError(f"cuLaunchKernel failed with status {status}")
        
        return x_ptr
    
    def softmax(self, out_tensor, inp_tensor):
        N, OC = inp_tensor.shape

        if "softmax_kernel" in self.func_lib:
            softmax_tensor_fnc = self.func_lib["softmax_kernel"]
        else:
            cuda_kernel_file = os.path.join(self.cuda_kernel_dir, 'softmax_func.cu')
            def read_file_content(code_filename):
                with open(code_filename, 'r') as f: return f.read()
            softmax_tensor_fnc = self.load_func(read_file_content(cuda_kernel_file), "softmax_kernel")
            self.func_lib["softmax_kernel"] = softmax_tensor_fnc
        
        block_x, block_y, block_z = 256, 1, 1
        grid_x, grid_y, grid_z = N, 1, 1
        shared_mem_size = int(2 * block_x / 32 * 4)# 32 * sizeof(float) 

        kernelArgs = [ctypes.addressof(out_tensor.dt_ptr), ctypes.addressof(inp_tensor.dt_ptr), 
                    ctypes.cast(ctypes.pointer(ctypes.c_uint32(N)), ctypes.c_void_p),
                    ctypes.cast(ctypes.pointer(ctypes.c_uint32(OC)), ctypes.c_void_p)]            
        c_args = (ctypes.c_void_p * len(kernelArgs))(*kernelArgs)

        status = cuda.cuLaunchKernel(softmax_tensor_fnc, grid_x, grid_y, grid_z,                # grid dim
                                            block_x, block_y, block_z,                          # block dim
                                            shared_mem_size, stream := cuda.CUstream(),         # shared mem and stream
                                            c_args, None)
        if status != 0:
            raise RuntimeError(f"cuLaunchKernel failed with status {status}")
        
        return out_tensor
    
    def transpose(self, in_tensor, out_tensor):
        w, h = out_tensor.shape
        if "s_transpose" in self.func_lib:
            s_transpose_tensor_fnc = self.func_lib["s_transpose"]
        else:
            cuda_kernel_file = os.path.join(self.cuda_kernel_dir, 'transpose.cu')
            def read_file_content(code_filename):
                with open(code_filename, 'r') as f: return f.read()
            s_transpose_tensor_fnc = self.load_func(read_file_content(cuda_kernel_file), "s_transpose")
            self.func_lib["s_transpose"] = s_transpose_tensor_fnc
        
        tile_dim = 4
        block_x, block_y, block_z = tile_dim, tile_dim, 1
        grid_x, grid_y, grid_z = math.ceil((w)/block_x), math.ceil((h)/block_y), 1
        shared_mem_size = tile_dim * tile_dim 
        kernelArgs = [ctypes.addressof(in_tensor.dt_ptr), ctypes.addressof(out_tensor.dt_ptr), 
                    ctypes.cast(ctypes.pointer(ctypes.c_uint32(w)), ctypes.c_void_p),
                    ctypes.cast(ctypes.pointer(ctypes.c_uint32(h)), ctypes.c_void_p)]            
        c_args = (ctypes.c_void_p * len(kernelArgs))(*kernelArgs)

        status = cuda.cuLaunchKernel(s_transpose_tensor_fnc, grid_x, grid_y, grid_z,                # grid dim
                                            block_x, block_y, block_z,                          # block dim
                                            shared_mem_size, stream := cuda.CUstream(),         # shared mem and stream
                                            c_args, None)
        if status != 0:
            raise RuntimeError(f"cuLaunchKernel failed with status {status}")
        
        return out_tensor