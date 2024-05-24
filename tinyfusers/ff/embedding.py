import gc
import ctypes
import cupy as cp
import numpy as np
from tinygrad import Tensor
from tinygrad.nn import Embedding as TEmbedding

libc = ctypes.CDLL("libc.so.6")
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

def embedding(idx, vocab_sz, embed_sz, weight):
      idx = cp.asarray(idx)
      cp.cuda.Device().synchronize()
      if idx.size == 0:
          return cp.empty(idx.shape + (embed_sz,), dtype=weight.dtype)
      arange_shp = (1, 1, vocab_sz, 1)
      weight_shp = (1, vocab_sz, embed_sz)
      big_shp = idx.shape + (vocab_sz, embed_sz)
      arange = cp.arange(vocab_sz, dtype=idx.dtype).reshape(arange_shp)
      arange = cp.broadcast_to(arange, big_shp)
      idx = idx.reshape(idx.shape + (1, 1))
      idx = cp.broadcast_to(idx, big_shp)
      vals = weight.reshape(weight_shp)
      vals = cp.broadcast_to(vals, big_shp)
      mask = (arange == idx)
      #vals *= mask # Gives wrong results for ~11% of elements.
      vals = vals*mask
      c_sum = cp.sum(vals, axis=2)
      o_np = cp.asnumpy(c_sum)
      del c_sum, vals, mask, idx, arange
      gc.collect()
      mempool.free_all_blocks()
      pinned_mempool.free_all_blocks()
      #print(f"{cp._default_memory_pool}, {pinned_mempool.n_free_blocks()}, {mempool.total_bytes()}, {mempool.used_bytes()}")
      #print(f"Freed memory==1: {libc.malloc_trim(0)}")
      return o_np 

class Embedding:
    def __init__(self, vocab_size: int, embed_size: int):
        self.vocab_sz = vocab_size
        self.embed_sz = embed_size
        # Will fix the weight loading and remove the next line.
        #self.weight = cp.random.randn(vocab_size, embed_size, dtype=cp.float32) * cp.sqrt(2. / (vocab_size + embed_size), dtype=cp.float32)
        self.weight = Tensor.glorot_uniform(vocab_size, embed_size)
    def __call__(self, idx: Tensor) -> Tensor:
        emb_tg = TEmbedding(self.vocab_sz, self.embed_sz)
        emb_tg.weight = self.weight
        #emb_tg.weight = Tensor(cp.asnumpy(self.weight), device="cuda")
        o_tg = emb_tg(idx)
        weight = cp.asarray(self.weight.numpy())
        # Temporary, for memory constraints. 
        o_p1 = embedding(idx.numpy()[0][:35].reshape(1, 35), self.vocab_sz, self.embed_sz, weight)
        o_p2 = embedding(idx.numpy()[0][35:].reshape(1, idx.shape[0]-35), self.vocab_sz, self.embed_sz, weight)
        #o_tf = embedding(idx.numpy(), self.vocab_sz, self.embed_sz, self.weight)
        o_tf = np.concatenate((o_p1[0], o_p2[0]), axis=0)
        o_tf = np.expand_dims(o_tf, axis=0)
        np.testing.assert_allclose(o_tf, o_tg.numpy(), atol=1e-2, rtol=1e-2)
        tensor_tf = Tensor(o_tf)
        #cp.cuda.Device().synchronize() # Is this really needed?
        return tensor_tf