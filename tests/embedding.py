import cupy as cp
from tinygrad.nn import Embedding as Emb
from tinygrad import Tensor
from tinyfusers.ff.embedding import Embedding

def test_embedding():
    vocab_size = 5
    embed_size = 5

    embedding = Embedding(vocab_size, embed_size)
    embedding1 = Emb(vocab_size, embed_size)
    embedding1.weight = Tensor(cp.asnumpy(embedding.weight), device="cuda")

    idx = cp.array([[1, 2], [1, 2]])
    embedded = embedding(idx)
    embedded1 = embedding1(Tensor(cp.asnumpy(idx)))
    cp.testing.assert_allclose(embedded, embedded1.numpy(), atol=1e-2, rtol=1e-2)

if __name__ == "__main__":
    test_embedding()