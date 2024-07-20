import numpy as np
import torch
from torch import nn
from tinyfusers.ff.embedding import embedding
from tinyfusers.storage.tensor import Tensor


def test_embedding():
    vocab_size, embed_size, n_dim = 10, 7, 10
    weight = np.random.randn(embed_size, vocab_size).astype(np.float32)
    x_np = np.zeros(shape=(embed_size, n_dim)).astype(np.float32)
    idx = np.random.randint(0, embed_size, size=(1, n_dim))
    _, N  = idx.shape

    emb  = nn.Embedding(vocab_size, embed_size)
    emb.weight = torch.nn.Parameter(torch.from_numpy(weight.T))

    for j, i in zip(idx[0], range(n_dim)):
        x_np[j][i] = 1

    x = Tensor.from_np(x_np).eval()
    w = Tensor.from_np(weight).eval()
    o_tf = embedding(w, x, embed_size).to('cpu').data.reshape(N, embed_size)
    o_t = emb(torch.from_numpy(idx.astype(np.int32)))

    np.testing.assert_allclose(o_t[0].detach().numpy(), o_tf)

if __name__ == "__main__":
    test_embedding()