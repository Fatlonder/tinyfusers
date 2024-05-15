import cudnn
import pytest
import torch
import cupy as cp
from torch import nn
import itertools
from looseversion import LooseVersion
from tinyfusers.ff.group_norm import group_norm
from tinyfusers.ff.group_norm import GroupNorm
from tinygrad import Tensor

embedding_dim_options = [768, 1024, 1280, 1600]
input_type_options = [torch.bfloat16, torch.float16]

all_options = [elem for elem in itertools.product(*[embedding_dim_options, input_type_options])]

@pytest.fixture(params=all_options)
def param_extract(request):
  return request.param

@pytest.mark.skipif(LooseVersion(cudnn.backend_version_string()) < "8.9.5", reason="LN not supported below cudnn 8.9.5")
def test_groupnorm(param_extract):

    embedding_dim, input_type = param_extract
    if input_type == torch.bfloat16:
        atol, rtol = 0.125, 0.125
    else:
        atol, rtol = 1e-2, 1e-2
    batch_size, seq_size = 16, 128
    N,C,H,W = batch_size * seq_size, embedding_dim, 2, 2
    num_groups = 2
    x_gpu = torch.randn(N, C, H, W, requires_grad=False, device="cuda", dtype=input_type).to(memory_format=torch.channels_last)
    group_norm_pt = nn.GroupNorm(num_groups, C, device="cuda")
    group_norm_tf = GroupNorm(num_groups, C, group_norm_pt.eps)

    o_tff = group_norm(Tensor(x_gpu.cpu().numpy(), device="cuda"), num_groups, group_norm_pt.eps)
    o_tf = group_norm_tf(Tensor(x_gpu.cpu().numpy(), device="cuda"))
    o_pt = group_norm_pt(x_gpu)

    torch.testing.assert_close(o_pt, torch.from_numpy(cp.asnumpy(o_tff)).to("cuda"), atol=atol, rtol=rtol)
    print(f"{o_pt}\n\n{o_tf}")

if __name__ == "__main__":
    test_groupnorm((1600, torch.float32))