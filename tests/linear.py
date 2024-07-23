from time import monotonic
import torch
import cupy as cp
import numpy as np
from tinygrad import Tensor
from tinyfusers.ff.linear import linear, linear_cublas
from tinyfusers.storage.tensor import Tensor

B, M, N, K, i_dtype = 1, 10000, 20000, 1000, torch.float16
embedding_dim_options = [768, 1024, 1280, 1600]
input_type_options = [torch.bfloat16, torch.float16, torch.float32]

def test_speedup(B, M, N, K, i_dtype):
    mat1 = torch.randn(B, M, K, requires_grad=False, device="cuda", dtype=i_dtype)
    mat2 = torch.randn(1, K, N, requires_grad=False, device="cuda", dtype=i_dtype)
    bias1 = torch.randn(1, 1, N, requires_grad=False, device="cuda", dtype=i_dtype)

    npmat1 = mat1.squeeze().cpu().numpy()
    npmat2 = mat2.squeeze().cpu().numpy()
    npbias1 = bias1.squeeze().cpu().numpy()

    tmat1 = Tensor(npmat1, device="cuda")
    tmat2 = Tensor(npmat2, device="cuda")
    tbias1 = Tensor(npbias1, device="cuda")

    cpmat1 = cp.asarray(npmat1)
    cpmat2 = cp.asarray(npmat2)
    cpbias1 = cp.asarray(npbias1)

    start_time = monotonic()
    tmat3 = (tmat1 @ tmat2 + tbias1).numpy() 
    print(f"Tinygrad: {monotonic()-start_time}")

    start_time = monotonic()
    mat3 = torch.matmul(mat1.squeeze(), mat2.squeeze()) + bias1.squeeze()
    print(f"Torch: {monotonic()-start_time}")

    start_time = monotonic()
    fmat3 = torch.nn.functional.linear(mat1.squeeze(), mat2.squeeze().T, bias=bias1.squeeze())
    print(f"Torch F: {monotonic()-start_time}")

    start_time = monotonic()
    cmat3 = cp.dot(cpmat1, cpmat2) + cpbias1
    print(f"CuPy: {monotonic()-start_time}")

    start_time = monotonic()
    tfmat3 = linear(mat1, mat2, bias1)
    print(f"CUDNN: {monotonic()-start_time}")

    start_time = monotonic()
    tfmat4 = linear_cublas(Tensor.from_np(mat1).eval(), Tensor.from_np(mat2).eval(), Tensor.from_np(bias1).eval())
    print(f"cublas: {monotonic()-start_time}")

    torch.testing.assert_close(mat3, torch.from_numpy(tmat3).to("cuda"), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(mat3, torch.from_numpy(cp.asnumpy(cmat3)).to("cuda"), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(mat3, fmat3, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(mat3, torch.from_numpy(cp.asnumpy(tfmat3)).to("cuda"), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(mat3.squeeze(), torch.from_numpy(tfmat4.data.reshape(M, N).to("cuda")), atol=1e-2, rtol=1e-2)

def test_cublas(M, K, N):
    w_np = np.random.randn(M,K).astype(np.float32)
    x_np = np.random.randn(K,N).astype(np.float32)
    w = Tensor.from_np(w_np).eval()
    x = Tensor.from_np(x_np).eval()

    res = linear_cublas(w, x, None).to('cpu').data.reshape(N,M).T
    res_np = w_np@x_np

    np.testing.assert_allclose(res, res_np)

def test_cublas_torch(M, N, K):
    i_dtype = torch.float32
    torch.manual_seed(0)

    w = torch.randn(M, K, requires_grad=False, device="cuda", dtype=i_dtype)
    x = torch.randn(K, N, requires_grad=False, device="cuda", dtype=i_dtype)
    bias = torch.randn(1, N, requires_grad=False, device="cuda", dtype=i_dtype)

    w_t = Tensor.from_np(w.cpu().numpy()).eval()
    x_t = Tensor.from_np(x.cpu().numpy()).eval()
    bias_t = Tensor.from_np(bias.cpu().numpy()).eval()

    res_tf = torch.nn.functional.linear(w.squeeze(), x.T, bias=bias)
    res_t = torch.matmul(w, x) + bias
    res_cb = linear_cublas(w_t, x_t, bias_t).to('cpu').data.reshape(N, M).T

    torch.testing.assert_close(torch.from_numpy(res_cb).to("cuda"), res_t.squeeze())
    torch.testing.assert_close(torch.from_numpy(res_cb).to("cuda"), res_tf.squeeze())

if __name__ == "__main__":
    M, K, N = 4, 3, 50
    test_speedup(B, M, N, K, i_dtype)
    test_cublas(M, K, N)
    test_cublas_torch(M, K, N)