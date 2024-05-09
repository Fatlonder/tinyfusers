import cudnn
import pytest
import torch
import cupy as cp
import itertools
from looseversion import LooseVersion
from tinyfusers.ff.linear import linear

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

if __name__ == "__main__":
    test_linear((768, torch.float16))