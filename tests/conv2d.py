import cupy as cp
import numpy as np
import cudnn
import torch
from tinyfusers.vision.conv2d import conv_2d
from tinyfusers.vision.conv2d import Conv2d
from tinygrad.nn import Conv2d as tConv2d
from tinygrad import Tensor
from time import monotonic

handle = cudnn.create_handle()

def test_conv_fp16():
    N, C, H, W = 1, 2, 10000, 10000
    K, R, S = 1, 2, 2
    padding, stride, dilation = [0,0], [1,1], [1,1]
    #X = cp.random.randn(N, C, H, W).astype(cp.float16)
    #W = cp.random.randn(K, C, R, S).astype(cp.float16)
    X = torch.randn((N, C, H, W), device="cuda", dtype=torch.float32)
    W = torch.randn((K, C, R, S), device="cuda", dtype=torch.float32)
    start_time = monotonic()
    o_cp = conv_2d(X, W, padding, stride, dilation)
    tf_time = f"{monotonic()-start_time}"

    o_tg = torch.from_numpy(cp.asnumpy(o_cp)).to("cuda")
    start_time = monotonic()
    o_pt = torch.nn.functional.conv2d(X, W, padding=padding, 
                                            stride=stride, 
                                            dilation=dilation).to("cuda").to(torch.float32)
    pt_time = f"{monotonic()-start_time}"

    print(f"pt: {pt_time}, tf: {tf_time}")
    torch.testing.assert_close(o_tg, o_pt, atol=1e-2, rtol=1e-2)

def test_conv_vs_tinygrad():
    tinyfusers_m = Conv2d(1, 1, 2)
    tinygrad_m = tConv2d(1, 1, 2, bias=False)

    t_input = Tensor.rand(1, 2, 10000, 10000)
    t_kernel = Tensor.ones(1, 2, 2, 2)
    tinygrad_m.weight = t_kernel

    input = cp.asarray(t_input.numpy().astype(np.float16))
    kernel = cp.asarray(t_kernel.numpy().astype(np.float16))
    tinyfusers_m.weight = kernel

    start_time = monotonic()
    o_tg = tinygrad_m(t_input)
    o_tg = o_tg.numpy().astype(np.float16)
    tg_time = f"{monotonic()-start_time}"

    start_time = monotonic()
    o_tf = tinyfusers_m(input)
    o_tf = cp.asnumpy(o_tf)
    tf_time = f"{monotonic()-start_time}"

    print(f"tg: {tg_time}, tf: {tf_time}")
    np.testing.assert_allclose(o_tg, o_tf, atol=1e-2, rtol=1e-2)

if __name__ =="__main__":
    test_conv_fp16()
    test_conv_vs_tinygrad()