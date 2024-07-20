import cupy as cp
import numpy as np
from tinyfusers.ff.linear import linear_cublas
from tinyfusers.storage.tensor import Tensor

def embedding(weight, idx):
      res = linear_cublas(weight, idx, None)
      return res 

class Embedding:
    def __init__(self, vocab_size: int, embed_size: int):
        self.vocab_sz = vocab_size
        self.embed_sz = embed_size
        self.weight = cp.ones((vocab_size, embed_size), dtype=cp.float32)
    def __call__(self, idx):
        _, N  = idx.shape
        x_np = np.zeros(shape=(self.embed_sz, N), dtype=np.float32)
        
        for j, i in zip(idx[0], range(N)):
            x_np[j][i] = 1
        x = Tensor.from_np(x_np).eval()

        return embedding(self.weight, x)#.to('cpu').data.reshape(N, self.embed_sz)