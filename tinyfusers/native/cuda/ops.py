import ctypes

class Cuda:
    def __init__(self,):
        self.dll = ctypes.CDLL('libcuda.so')
    
    def cuCtxCreate_v2(self, pctx, flag, device_id):
        self.dll.cuCtxCreate_v2.restype = ctypes.c_int
        self.dll.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUctx_st)), ctypes.c_uint32, ctypes.c_int32]
        status = self.dll.cuCtxCreate_v2(pctx, flag, device_id)
        return status
    
    def cuModuleLoadData(self, module, data):
        self.dll.cuModuleLoadData.restype = ctypes.c_int
        self.dll.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmod_st)), ctypes.POINTER(None)]
        status = self.dll.cuModuleLoadData(ctypes.byref(module), data)
        return status
    
    def cuModuleGetFunction(self, func, module, f_name):
        self.dll.cuModuleGetFunction.restype = ctypes.c_int
        self.dll.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_cuFunction)), ctypes.POINTER(struct_CUmod_st), ctypes.POINTER(ctypes.c_char)]
        status = self.dll.cuModuleGetFunction(ctypes.byref(func), module, f_name)
        return status
    
    def cuLaunchKernel(self, func_p, grid_x, grid_y, grid_z, block_x, block_y, block_z, sharedMemBytes, hStream, kernelParams, extra):
        self.dll.cuLaunchKernel.restype = ctypes.c_int
        self.dll.cuLaunchKernel.argtypes = [ctypes.POINTER(struct_cuFunction), 
                                    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, 
                                    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, 
                                    ctypes.c_uint32, CUstream, 
                                    ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.POINTER(None))]
        status = self.dll.cuLaunchKernel(func_p, grid_x, grid_y, grid_z, block_x, block_y, block_z, sharedMemBytes, hStream, kernelParams, extra)
        return status
    
class Cudart:
    def __init__(self,):
        self.dll = ctypes.CDLL('libcudart.so')

    def cudaMalloc(self, a:ctypes.c_void_p, nbytes):
        self.dll.cudaMalloc.restype = ctypes.c_int
        self.dll.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        status = self.dll.cudaMalloc(ctypes.byref(a), ctypes.c_size_t(nbytes))
        return status
    
    def cudaMemcpy(self, dst, src, nbytes, type: int):
        self.dll.cudaMemcpy.restype = ctypes.c_int
        self.dll.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        status = self.dll.cudaMemcpy(dst, src, ctypes.c_size_t(nbytes), type)
        return status    
    
    def cudaFree(self, devPtr: ctypes.c_void_p):
        self.dll.cudaFree.restype = ctypes.c_int
        self.dll.cudaFree.argtypes = [ctypes.c_void_p]
        status = self.dll.cudaFree(devPtr)
        return status 
    
    def cudaDeviceGetAttribute(self, atr_ptr: ctypes.c_void_p, attr, device_id):
        self.dll.cudaDeviceGetAttribute.restype = ctypes.c_int
        self.dll.cudaDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32]
        status = self.dll.cudaDeviceGetAttribute(atr_ptr, attr, device_id)
        return status 
     
class struct_CUmod_st(ctypes.Structure):
    pass
class struct_CUctx_st(ctypes.Structure):
    pass
class struct_cuFunction(ctypes.Structure):
    pass
class struct_CUstream_st(ctypes.Structure):
    pass

CUmodule = ctypes.POINTER(struct_CUmod_st)
CUcontext = ctypes.POINTER(struct_CUctx_st)
CUfunction = ctypes.POINTER(struct_cuFunction)
CUstream = ctypes.POINTER(struct_CUstream_st)

cuda = Cuda()
cudart = Cudart()

cuda.CUmodule = CUmodule
cuda.CUcontext = CUcontext
cuda.CUfunction = CUfunction
cuda.CUstream = CUstream
cuda.CU_CTX_SCHED_AUTO = 0

cudart.CUDA_SUCCESS = 0
cudart.cudaMemcpyHostToDevice = 1
cudart.cudaMemcpyDeviceToHost = 2
cudart.cudaDevAttrComputeCapabilityMajor = 75
cudart.cudaDevAttrComputeCapabilityMinor = 76