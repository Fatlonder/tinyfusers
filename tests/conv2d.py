import cupy as cp
import cudnn
import torch
from ..native.conv2d_graph import conv_2d

handle = cudnn.create_handle()

def test_conv_fp16():
    N, C, H, W = 1, 1, 3, 3
    K, R, S = 1, 2, 2
    padding, stride, dilation = [0,0], [1,1], [1,1]
    #X = cp.random.randn(N, C, H, W)
    #W = cp.random.randn(K, C, R, S)
    X = torch.randn((N, C, H, W), device="cuda", dtype=torch.float16)
    W = torch.randn((K, C, R, S), device="cuda", dtype=torch.float16)
    Y_cp = conv_2d(X, W, padding, stride, dilation)
    Y = torch.from_numpy(cp.asnumpy(Y_cp))
    Y_expected = torch.nn.functional.conv2d(X, W, padding=padding, 
                                            stride=stride, 
                                            dilation=dilation).to("cuda").to(torch.float16)
    torch.testing.assert_close(Y, Y_expected, atol=1e-2, rtol=1e-2)


if __name__ =="__main__":
    test_conv_fp16()