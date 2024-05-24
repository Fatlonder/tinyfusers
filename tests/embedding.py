import cupy as cp
from tinygrad.nn import Embedding as Emb
from tinygrad import Tensor
from tinyfusers.ff.embedding import Embedding

def test_embedding():
    vocab_size, embed_size = 49408, 768
    n_dim = 77

    embedding = Embedding(vocab_size, embed_size)
    embedding1 = Emb(vocab_size, embed_size)
    embedding1.weight = Tensor(cp.asnumpy(embedding.weight), device="cuda")

    idx = cp.array([[1, 2], [1, 2]])
    #idx = np.arange(n_dim, dtype=np.int16).reshape(1,n_dim)
    t_idx = Tensor(cp.asnumpy(idx))
    embedded = embedding(t_idx)
    embedded1 = embedding1(t_idx)
    cp.testing.assert_allclose(embedded, embedded1.numpy(), atol=1e-2, rtol=1e-2)

if __name__ == "__main__":
    test_embedding()