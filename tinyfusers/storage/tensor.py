import cupy as cp
import functools
from .device import Device

class Tensor:
    def __init__(self, device: Device):
        self.device = device if device else Tensor.default_device()
    
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