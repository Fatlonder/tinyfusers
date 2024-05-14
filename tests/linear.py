import cudnn
import pytest
import torch
import cupy as cp
import itertools
import torch
from tinygrad import Tensor
from time import monotonic
from tinyfusers.ff.linear import linear
from looseversion import LooseVersion
from tinyfusers.ff.linear import linear

B, M, N, K, i_dtype = 1, 10000, 20000, 1000, torch.float16
embedding_dim_options = [768, 1024, 1280, 1600]
input_type_options = [torch.bfloat16, torch.float16]

all_options = [elem for elem in itertools.product(*[embedding_dim_options, input_type_options])]

@pytest.fixture(params=all_options)
def param_extract(request):
  return request.param

@pytest.mark.skipif(LooseVersion(cudnn.backend_version_string()) < "8.9.5", reason="LN not supported below cudnn 8.9.5")
def test_linear(param_extract):
    embedding_dim, input_type = param_extract
    if input_type == torch.bfloat16:
        atol, rtol = 0.125, 0.125
    else:
        atol, rtol = 1e-2, 1e-2
    batch_size = 16
    (b, m, n, k), input_type = ((1, batch_size, embedding_dim, embedding_dim*4), input_type)
    X_gpu = torch.randn(b, m, n, requires_grad=False, device="cuda", dtype=input_type)
    W_gpu = torch.randn(1, n, k, requires_grad=False, device="cuda", dtype=input_type)
    B_gpu = torch.randn(1, 1, k, requires_grad=False, device="cuda", dtype=input_type)

    Y_expected = torch.nn.functional.linear(X_gpu, W_gpu.squeeze().T, bias=B_gpu.squeeze())
    Y_actual = linear(X_gpu, W_gpu, B_gpu)

    torch.testing.assert_close(Y_expected[0], torch.from_numpy(cp.asnumpy(Y_actual)).to("cuda"), atol=atol, rtol=rtol)

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

    torch.testing.assert_close(mat3, torch.from_numpy(tmat3).to("cuda"), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(mat3, torch.from_numpy(cp.asnumpy(cmat3)).to("cuda"), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(mat3, fmat3, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(mat3, torch.from_numpy(cp.asnumpy(tfmat3)).to("cuda"), atol=1e-2, rtol=1e-2)

if __name__ == "__main__":
    test_linear((768, torch.float16))
    test_speedup(B, M, N, K, i_dtype)