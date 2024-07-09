import ctypes

cublasHandle_t = ctypes.c_void_p

class Cublas:
    def __init__(self,):
        self.dll = ctypes.CDLL('libcublas.so.12')

    def cublasCreate(self, handle):
        self.dll.cublasCreate_v2.restype = ctypes.c_int
        self.dll.cublasCreate_v2.argtypes = [ctypes.POINTER(cublasHandle_t)]

        return self.dll.cublasCreate_v2(ctypes.byref(handle))
    
    def cublasDestroy(self, handle):
        self.dll.cublasDestroy_v2.restype = ctypes.c_int
        self.dll.cublasDestroy_v2.argtypes = [cublasHandle_t]
    
        return self.dll.cublasDestroy_v2(handle)
    
    def cublasSgemm(self, handle, transa, transb,  m, n, k, alpha, d_a, lda, d_b, ldb, beta, d_c, ldc):
        self.dll.cublasSgemm_v2.restype = ctypes.c_int
        self.dll.cublasSgemm_v2.argtypes = [cublasHandle_t, ctypes.c_uint, ctypes.c_uint, 
                                  ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, 
                                  ctypes.POINTER(ctypes.c_float), 
                                  ctypes.POINTER(ctypes.c_float), ctypes.c_int32, 
                                  ctypes.POINTER(ctypes.c_float), ctypes.c_int32, 
                                  ctypes.POINTER(ctypes.c_float), 
                                  ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
        d_a_float_p = ctypes.cast(d_a, ctypes.POINTER(ctypes.c_float))
        d_b_float_p = ctypes.cast(d_b, ctypes.POINTER(ctypes.c_float))
        d_c_float_p = ctypes.cast(d_c, ctypes.POINTER(ctypes.c_float))
        status = self.dll.cublasSgemm_v2(handle, transa, transb,  m, n, k, 
                               ctypes.byref(ctypes.c_float(alpha)), 
                               d_a_float_p, lda, 
                               d_b_float_p, ldb, 
                               ctypes.byref(ctypes.c_float(beta)), 
                               d_c_float_p, ldc)
    
        return status

cublas = Cublas()
cublas.CUBLAS_STATUS_SUCCESS = 0
cublas.CUBLAS_STATUS_NOT_INITIALIZED = 1
cublas.CUBLAS_STATUS_ALLOC_FAILED = 2
cublas.CUBLAS_STATUS_INVALID_VALUE = 3
cublas.CUBLAS_STATUS_ARCH_MISMATCH = 4
cublas.CUBLAS_STATUS_MAPPING_ERROR = 5
cublas.CUBLAS_STATUS_EXECUTION_FAILED = 6
cublas.CUBLAS_STATUS_INTERNAL_ERROR = 7
cublas.CUBLAS_STATUS_NOT_SUPPORTED = 8
cublas.CUBLAS_STATUS_LICENSE_ERROR = 9
cublas.CUBLAS_OP_N = 0
cublas.CUBLAS_OP_T = 1
cublas.CUBLAS_OP_C = 2
cublas.cublasHandle_t = ctypes.c_void_p
