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
    
    def load_library_from_path(self, device, lib_path, lib_name = "cublas"):
        return 0

    def __str__(self):
        return str(self.device)
    
    def load_func(self, code_str, func_name):
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