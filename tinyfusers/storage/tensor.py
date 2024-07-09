import math
import cupy as cp
import numpy as np
import ctypes
import functools
from .device import Device
from ..native.cuda.ops import cudart

class Tensor:
    def __init__(self, shape:tuple, dtype = np.float32, device: Device = "cuda", data: np.array = None):
        self.device = device if device else Tensor.default_device()
        self.dt_ptr = ctypes.c_void_p()
        self.shape = shape if data is None else data.shape
        self.dtype = dtype if data is None else data.dtype
        self.num_elem = math.prod(self.shape)
        self.data = data

    def eval(self):
        if self.device == "cuda":
            status = cudart.cudaMalloc(self.dt_ptr, self.num_elem * np.dtype(self.dtype).itemsize)
            if status != cudart.CUDA_SUCCESS:
                raise RuntimeError('cudaMalloc failed with status {}'.format(status))
            if self.data is not None:
                status = cudart.cudaMemcpy(self.dt_ptr, self.data.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p)), 
                                       self.data.nbytes, cudart.cudaMemcpyHostToDevice)
                if status != cudart.CUDA_SUCCESS:
                    raise RuntimeError('cudaMemcpy (Host to Device) failed with status {}'.format(status))
        else:
            self.data = np.zeros(self.shape).astype(self.dtype)
            self.stride = self.data.strides

    def to(self, device):
        if self.device == device: return self
        if device == "cpu":
            status = cudart.cudaMemcpy(self.data.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p)), 
                                       self.dt_ptr, self.data.nbytes, cudart.cudaMemcpyDeviceToHost)
            if status != cudart.CUDA_SUCCESS:
                raise RuntimeError('cudaMemcpy (Host to Device) failed with status {}'.format(status))
            self.dta_ptr = 0
        return self

    @staticmethod
    def from_np(data: np.array):
        return Tensor(data.shape, data.dtype, device="cuda", data=data)

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