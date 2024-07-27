import math
import cupy as cp
import numpy as np
import ctypes
import functools
from .device import Device
from ..native.cuda.ops import cudart

class Tensor:
    def __init__(self, shape:tuple, dtype = np.float32, device: Device = None, data: np.array = None):
        self.device = device if device else Tensor.default_device()
        self.dt_ptr = ctypes.c_void_p()
        self.shape = shape if data is None else data.shape
        self.dtype = dtype if data is None else data.dtype
        self.num_elem = math.prod(self.shape)
        self.data = data
        self.nbytes = self.num_elem * np.dtype(self.dtype).itemsize
        self.strides = None

    def eval(self):
        if str(self.device) == "cuda" and self.dt_ptr != 0:
            status = cudart.cudaMalloc(self.dt_ptr, self.nbytes)
            if status != cudart.CUDA_SUCCESS:
                raise RuntimeError('cudaMalloc failed with status {}'.format(status))
            if self.data is not None:
                status = cudart.cudaMemcpy(self.dt_ptr, self.data.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p)), 
                                       self.data.nbytes, cudart.cudaMemcpyHostToDevice)
                if status != cudart.CUDA_SUCCESS:
                    raise RuntimeError('cudaMemcpy (Host to Device) failed with status {}'.format(status))
            return self
        else:
            self.data = np.zeros(self.shape).astype(self.dtype)
            self.stride = self.data.strides
            return self

    def to(self, device):
        if self.device == device: return self
        if device == "cpu":
            status = cudart.cudaMemcpy(self.data.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p)), 
                                       self.dt_ptr, self.data.nbytes, cudart.cudaMemcpyDeviceToHost)
            if status != cudart.CUDA_SUCCESS:
                raise RuntimeError('cudaMemcpy (Host to Device) failed with status {}'.format(status))
            
            status = cudart.cudaFree(self.dt_ptr)
            if status != cudart.CUDA_SUCCESS:
                raise RuntimeError('cudaFree failed with status {}'.format(status))
            
            self.dta_ptr = 0
            self.device = Device("cpu")
        return self
    
    @staticmethod
    def from_np(data: np.array):
        return Tensor(data.shape, data.dtype, device=None, data=data)
    
    @staticmethod
    def zeros(shape, dtype):
        return Tensor.from_np(np.zeros(shape, dtype=dtype))
    
    @staticmethod
    def default_device():
        return Device("cuda")
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + cp.exp(-x))
    
    @staticmethod
    def silu(x):
        return x * Tensor.sigmoid(x)
    
    @staticmethod
    def sequential(iterable, init):
        return functools.reduce(lambda x, f: f(x), iterable, init)
    
    @staticmethod
    def quick_gelu(x):
        return x * Tensor.sigmoid(x * 1.702)
    
    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + cp.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))
    
    @staticmethod
    def swish(x):
        return x * Tensor.sigmoid(x)
    
    def __add__(self, other):
        raise RuntimeError(f"Tensor.__add__ not implemented")