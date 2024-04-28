import cupy as cp
import numpy as np
import cudnn
import torch
from tinyfusers.vision.conv2d import Conv2d
from tinygrad.nn import Conv2d as tConv2d
from tinygrad import Tensor
from time import monotonic

handle = cudnn.create_handle()

def test_conv_fp16():
    N, C, H, W = 1, 1, 3, 3
    K, R, S = 1, 2, 2
    padding, stride, dilation = [0,0], [1,1], [1,1]
    #X = cp.random.randn(N, C, H, W)
    #W = cp.random.randn(K, C, R, S)
    X = torch.randn((N, C, H, W), device="cuda", dtype=torch.float16)
    W = torch.randn((K, C, R, S), device="cuda", dtype=torch.float16)
    Y_cp = Conv2d(X, W, padding, stride, dilation)
    Y = torch.from_numpy(cp.asnumpy(Y_cp))
    Y_expected = torch.nn.functional.conv2d(X, W, padding=padding, 
                                            stride=stride, 
                                            dilation=dilation).to("cuda").to(torch.float16)
    torch.testing.assert_close(Y, Y_expected, atol=1e-2, rtol=1e-2)

def test_conv_vs_tinygrad():
    tinyfusers_m = Conv2d(1, 1, 2)
    tinygrad_m = tConv2d(1, 1, 2, bias=False)

    t_input = Tensor.rand(1, 1, 30000, 30000)#DLPack interface?
    t_kernel = Tensor.ones(1, 1, 2, 2)
    tinygrad_m.weight = t_kernel

    input = cp.asarray(t_input.numpy().astype(np.float16))
    kernel = cp.asarray(t_kernel.numpy().astype(np.float16))
    tinyfusers_m.weight = kernel

    start_time = monotonic()
    t_output = tinygrad_m(t_input)
    t_output = t_output.numpy()
    print(f"{monotonic()-start_time}\n\n")
    print(f"{input.shape}, {t_output.shape}\n\n{t_output}")

    start_time = monotonic()
    output = tinyfusers_m(input)
    output = cp.asnumpy(output)
    print(f"{monotonic()-start_time}\n\n")
    print(f"{input.shape}, {output.shape}\n\n{output}")

if __name__ =="__main__":
    test_conv_fp16()