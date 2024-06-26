import cudnn
import pytest
import torch
import cupy as cp
import itertools
from looseversion import LooseVersion
from tinyfusers.ff.layer_norm import layer_norm
from tinygrad.nn import LayerNorm
from tinygrad import Tensor
from typing import Union, Tuple

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

    x_gpu = torch.randn(N, C, H, W, requires_grad=False, device="cuda", dtype=input_type)
    scale_gpu = torch.randn(1, C, H, W, requires_grad=False, device="cuda", dtype=input_type)
    bias_gpu = torch.randn(1, C, H, W, requires_grad=False, device="cuda", dtype=input_type)
    epsilon_cpu = torch.full((1, 1, 1, 1), epsilon_value, requires_grad=False, device="cpu", dtype=torch.float32)

    o_pt = torch.nn.functional.layer_norm(x_gpu, [C, H, W], weight=scale_gpu.squeeze(0), bias=bias_gpu.squeeze(0), eps=epsilon_value)
    o_tf = layer_norm(x_gpu, scale_gpu, bias_gpu, epsilon_cpu)

    torch.testing.assert_close(o_pt, torch.from_numpy(cp.asnumpy(o_tf)).to("cuda"), atol=atol, rtol=rtol)

@pytest.mark.skipif(LooseVersion(cudnn.backend_version_string()) < "8.9.5", reason="LN not supported below cudnn 8.9.5")
def test_layernorm_tinygrad():
    input_type = torch.float32
    atol, rtol = 1e-2, 1e-2
    batch_size, seq_size = 16, 128
    N,C,H,W = 10, 32, 10, 10
    epsilon_value = 1e-3

    x_gpu = torch.randn(N, C, H, W, requires_grad=False, device="cuda", dtype=input_type)
    scale_gpu = torch.randn(1, 1, 1, W, requires_grad=False, device="cuda", dtype=input_type)
    bias_gpu = torch.randn(1, 1, 1, W, requires_grad=False, device="cuda", dtype=input_type)
    epsilon_cpu = torch.full((1, 1, 1, 1), epsilon_value, requires_grad=False, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        torch_layer = torch.nn.LayerNorm([W], eps=epsilon_value, device="cuda").eval()
        torch_layer.weight[:] = scale_gpu
        torch_layer.bias[:] = bias_gpu

    ln = LayerNorm(W, eps=epsilon_value)
    ln.weight = Tensor(scale_gpu.cpu().numpy())
    ln.bias = Tensor(bias_gpu.cpu().numpy())
    ln.eps = Tensor(epsilon_cpu.numpy())

    #o_pt = torch.nn.functional.layer_norm(x_gpu, [W], weight=scale_gpu.squeeze(0), bias=bias_gpu.squeeze(0), eps=epsilon_value)
    o_pt = torch_layer(x_gpu)
    o_tf = layer_norm(x_gpu, scale_gpu, bias_gpu, epsilon_cpu)
    o_tg = ln(Tensor(x_gpu.squeeze(0).cpu().numpy()))
    print(f"{o_tg.shape}, {o_pt.shape}, {o_tf.shape}")
    torch.testing.assert_close(o_pt, torch.from_numpy(cp.asnumpy(o_tf)).to("cuda"), atol=atol, rtol=rtol)
    torch.testing.assert_close(o_pt, torch.from_numpy(cp.asnumpy(o_tg.numpy())).to("cuda"), atol=atol, rtol=rtol)
if __name__ == "__main__":
    test_layernorm((1600, torch.bfloat16))#float32
    test_layernorm_tinygrad()