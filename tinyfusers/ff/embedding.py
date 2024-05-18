import cupy as cp
from tinygrad import Tensor

class Embedding:
    def __init__(self, vocab_size: int, embed_size: int):
        self.vocab_sz = vocab_size
        self.embed_sz = embed_size
        self.weight = cp.random.randn(vocab_size, embed_size) * cp.sqrt(2. / (vocab_size + embed_size))
    def __call__(self, idx: cp.ndarray) -> cp.ndarray:
        if idx.size == 0:
            return cp.empty(idx.shape + (self.embed_sz,), dtype=self.weight.dtype)
        arange_shp = (1, 1, self.vocab_sz, 1)
        weight_shp = (1, self.vocab_sz, self.embed_sz)
        big_shp = idx.shape + (self.vocab_sz, self.embed_sz)
        if not hasattr(self, 'arange'):
            self.arange = cp.arange(self.vocab_sz, dtype=idx.dtype).reshape(arange_shp)
        arange = cp.broadcast_to(self.arange, big_shp)
        idx = idx.reshape(idx.shape + (1, 1))
        idx = cp.broadcast_to(idx, big_shp)
        vals = self.weight.reshape(weight_shp)
        vals = cp.broadcast_to(vals, big_shp)
        return Tensor(cp.asnumpy(cp.sum(cp.multiply(cp.equal(arange, idx), vals), axis=2)), device="cuda")