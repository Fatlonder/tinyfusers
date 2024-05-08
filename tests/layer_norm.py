import cudnn
import pytest
import torch
import itertools
from looseversion import LooseVersion
from tinyfusers.ff.layer_norm import layer_norm


embedding_dim_options = [768, 1024, 1280, 1600]
input_type_options = [torch.bfloat16, torch.float16]

all_options = [elem for elem in itertools.product(*[embedding_dim_options, input_type_options])]

@pytest.fixture(params=all_options)
def param_extract(request):
  return request.param

@pytest.mark.skipif(LooseVersion(cudnn.backend_version_string()) < "8.9.5", reason="LN not supported below cudnn 8.9.5")
def test_layernorm(param_extract):

    embedding_dim, input_type = param_extract
    if input_type == torch.bfloat16:
        atol, rtol = 0.125, 0.125
    else:
        atol, rtol = 1e-2, 1e-2
    batch_size, seq_size = 16, 128
    N,C,H,W = batch_size * seq_size, embedding_dim, 10, 10
    epsilon_value = 1e-3
    x_gpu = 3*torch.randn(N, C, H, W, requires_grad=True, device="cuda", dtype=input_type).to(memory_format=torch.channels_last) - 0.5
    scale_gpu = 5*torch.randn(1, C, H, W, requires_grad=True, device="cuda", dtype=input_type).to(memory_format=torch.channels_last) - 1
    bias_gpu = 7*torch.randn(1, C, H, W, requires_grad=True, device="cuda", dtype=input_type).to(memory_format=torch.channels_last) -2
    epsilon_cpu = torch.full((1, 1, 1, 1), epsilon_value, requires_grad=False, device="cpu", dtype=torch.float32)
    Y_expected = torch.nn.functional.layer_norm(x_gpu, [C, H, W], weight=scale_gpu.squeeze(0), bias=bias_gpu.squeeze(0), eps=epsilon_value)
    Y_actual = layer_norm(x_gpu, scale_gpu, bias_gpu, epsilon_cpu)
    torch.testing.assert_close(Y_expected, torch.from_numpy(Y_actual).to("cuda"), atol=atol, rtol=rtol)

if __name__ == "__main__":
    test_layernorm((1600, torch.bfloat16))#float32