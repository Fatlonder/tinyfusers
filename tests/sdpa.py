import math
import cudnn
import cupy as cp
import torch 
from tinygrad import Tensor
from time import monotonic
import os
from looseversion import LooseVersion
import pytest
from tinyfusers.attention.sdpa import scaled_dot_product_attention


B = 35 # batch size
T = 1024 # maximum sequence length
C = 768 
NH = 12 # query number of heads
HS = int(C / NH) # embedding dimension per head
scale = cp.single(1.0 / math.sqrt(HS))
N = T
dims = (B, NH, T, HS)
input_type = torch.float32
cuda_dir = os.path.abspath('tinyfusers/native/cuda/')
loaded_from_source = os.path.join(cuda_dir, 'softmax.cu')
options=('--use_fast_math', '-lcublas -lcublasLt', '-D__CUDA_NO_HALF_CONVERSIONS__', f"-I{cuda_dir}")

with open(loaded_from_source, 'r') as f:
    code = f.read()

module = cp.RawModule(code=code, backend='nvcc', options=options) 
scale_kernel = module.get_function('scale_kernel')
softmax_forward_kernel = module.get_function('softmax_forward_kernel')

@pytest.mark.skipif(LooseVersion(cudnn.backend_version_string()) < "8.9.5", reason="LN not supported below cudnn 8.9.5")
def test_precision(B, T, C, NH, HS, scale, dims, input_type):
    softmax_block_size = 256;
    grid_size = B * NH * T;
    shared_mem_size = 2 * softmax_block_size / 32 * 4 #sizeof(float);

    att = cp.zeros((B * NH * T, T), dtype=cp.float32)
    preatt = cp.zeros((B, NH, T, T), dtype=cp.float32)

    q_gpu = torch.randn((B, NH, T, HS), requires_grad=False, device="cuda", dtype=input_type)
    k_gpu = torch.randn((B, NH, T, HS), requires_grad=False, device="cuda", dtype=input_type)
    v_gpu = torch.randn((B, NH, T, HS), requires_grad=False, device="cuda", dtype=input_type)

    q_np = q_gpu.cpu().numpy()
    k_np = k_gpu.cpu().numpy()
    v_np = v_gpu.cpu().numpy()

    q = cp.asarray(q_np)
    k = cp.asarray(k_np)
    v = cp.asarray(v_np)

    q_t = Tensor(q_np)
    k_t = Tensor(k_np)
    v_t = Tensor(v_np)

    start_time = monotonic()
    preatt =  cp.matmul(q, cp.transpose(k, axes=(0,1,3,2)))
    scale_kernel((N,), (N,), (preatt, scale, B, NH, T))
    softmax_forward_kernel(grid=(grid_size,), block=(softmax_block_size,), args=(att, preatt, B * NH * T, T), shared_mem=shared_mem_size)
    o_tf = cp.matmul(cp.reshape(att, (B, NH, T, T)), v)
    cp_time = monotonic()-start_time

    start_time = monotonic()
    o_pt = torch.nn.functional.scaled_dot_product_attention(q_gpu, k_gpu, v_gpu, scale=scale)
    pt_time = monotonic()-start_time

    start_time = monotonic()
    o_tg = Tensor.scaled_dot_product_attention(q_t, k_t, v_t).realize()
    tg_time = monotonic()-start_time

    start_time = monotonic()
    o_tff = scaled_dot_product_attention(q_t, k_t, v_t)
    tff_time = monotonic()-start_time

    print(f"TF: {cp_time}, PT: {pt_time}, TG: {tg_time}, Tff: {tff_time}")

    torch.testing.assert_close(torch.from_numpy(cp.asnumpy(o_tf)).to("cuda"), o_pt, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(o_pt, torch.from_numpy(o_tg.numpy()).to("cuda"), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(o_pt, torch.from_numpy(o_tff.numpy()).to("cuda"), atol=1e-2, rtol=1e-2)

@pytest.mark.skipif(LooseVersion(cudnn.backend_version_string()) < "8.9.5", reason="LN not supported below cudnn 8.9.5")
def test_scaled_dot_product_attention(scale, dims, input_type):
    q_gpu = torch.randn(dims, requires_grad=False, device="cuda", dtype=input_type)
    k_gpu = torch.randn(dims, requires_grad=False, device="cuda", dtype=input_type)
    v_gpu = torch.randn(dims, requires_grad=False, device="cuda", dtype=input_type)

    q_np = q_gpu.cpu().numpy()
    k_np = k_gpu.cpu().numpy()
    v_np = v_gpu.cpu().numpy()

    q = cp.asarray(q_np)
    k = cp.asarray(k_np)
    v = cp.asarray(v_np)

    o_tf = scaled_dot_product_attention(q, k, v)
    o_pt = torch.nn.functional.scaled_dot_product_attention(q_gpu, k_gpu, v_gpu, scale=scale)

    torch.testing.assert_close(torch.from_numpy(cp.asnumpy(o_tf)).to("cuda"), o_pt, atol=1e-2, rtol=1e-2)

if __name__ == "__main__":
    test_precision(B, T, C, NH, HS, scale, N, dims, input_type)
    test_scaled_dot_product_attention(scale, dims, input_type)

