import gc
import ctypes
import cupy as cp

libc = ctypes.CDLL("libc.so.6")
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

def embedding(idx, vocab_sz, embed_sz, weight):
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
      del vals, mask, idx, arange #, c_sum
      gc.collect()
      mempool.free_all_blocks()
      pinned_mempool.free_all_blocks()
      #print(f"{cp._default_memory_pool}, {pinned_mempool.n_free_blocks()}, {mempool.total_bytes()}, {mempool.used_bytes()}")
      #print(f"Freed memory==1: {libc.malloc_trim(0)}")
      return c_sum 

class Embedding:
    def __init__(self, vocab_size: int, embed_size: int):
        self.vocab_sz = vocab_size
        self.embed_sz = embed_size
        self.weight = cp.ones((vocab_size, embed_size), dtype=cp.float32)
    def __call__(self, idx):
        # Temporary, for memory constraints. 
        o_p1 = embedding(idx[0][:35].reshape(1, 35), self.vocab_sz, self.embed_sz, self.weight)
        o_p2 = embedding(idx[0][35:].reshape(1, idx.shape[0]-35), self.vocab_sz, self.embed_sz, self.weight)
        o_tf = cp.concatenate((o_p1[0], o_p2[0]), axis=0)
        o_tf = cp.expand_dims(o_tf, axis=0)
        return o_tf