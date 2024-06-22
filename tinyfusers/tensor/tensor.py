import cupy as cp
import functools

class Tensor:
    def __init__(self):
        pass
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + cp.exp(-x))
    
    @staticmethod
    def silu(x):
        return x * Tensor.sigmoid(x)
    
    @staticmethod
    def sequential(iterable, init):
        return functools.reduce(lambda x, f: f(x), iterable, init)