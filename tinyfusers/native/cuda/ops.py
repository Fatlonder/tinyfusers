import ctypes

CUDA_SUCCESS = 0
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2

class Cuda:
    def __init__(self,):
        self.dll = ctypes.CDLL(ctypes.util.find_library('libcuba.so'))

    def cudaMalloc(self, a:ctypes.c_void_p, nbytes):
        self.dll.cudaMalloc.restype = ctypes.c_int
        self.dll.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        status = cudart.cudaMalloc(ctypes.byref(a), ctypes.c_size_t(nbytes))
        return status
    
    def cudaMemcpy(self, dst, src, nbytes, type):
        self.dll.cudaMemcpy.restype = ctypes.c_int
        self.dll.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        status = self.dll.cudaMemcpy(dst, src, ctypes.c_size_t(nbytes), type)
        return status    
    
    def cudaFree(self, devPtr: ctypes.c_void_p):
        self.dll.cudaFree.restype = ctypes.c_int
        self.dll.cudaFree.argtypes = [ctypes.c_void_p]
        status = self.dll.cudaFree(devPtr)
        return status 
     
cudart = Cuda()