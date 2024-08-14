import cupy as cp
import os
import numpy as np
from tinyfusers import Tensor
from tinyfusers.storage.device import Device

def test_scale_tensor(inp, scaler, B, T, OC, NH):
    device = Device("cuda")
    inp_scaled = scaler*inp
    tt = Tensor.from_np(inp).eval()
    device.scale_tensor(tt.dt_ptr, scaler, B, T, NH, OC)    
    np.testing.assert_allclose(inp_scaled, tt.to('cpu').data)

def test_softmax(inp_tensor, out_tensor):
    device = Device("cuda")
    input_type = np.float32
    B, NH, T, HS = 1, 2, 2, 2

    q_ = np.random.rand(B * NH * T, HS).astype(input_type)
    inp_tensor = Tensor.from_np(q_).eval()
    out_tensor = Tensor.zeros(q_.shape, np.float32).eval()

    device.softmax(inp_tensor, out_tensor) 
    probs = out_tensor.to('cpu').data

def test_softmax_v_cp():
    device = Device("cuda")
    input_type = np.float32
    cuda_dir = os.path.abspath('tinyfusers/tinyfusers/native/cuda/')
    loaded_from_source = os.path.join(cuda_dir, 'softmax.cu')
    def read_code(code_filename):
        with open(code_filename, 'r') as f:
            code = f.read()
            return code
    options=('--use_fast_math', '-lcublas -lcublasLt', '-D__CUDA_NO_HALF_CONVERSIONS__', f"-I{cuda_dir}")
    module = cp.RawModule(code=read_code(loaded_from_source), backend='nvcc', options=options) 
    softmax_forward_kernel = module.get_function('softmax_kernel')
    cur_stream = cp.cuda.get_current_stream()
    cur_stream.use()
    B, NH, T_q, T_k = 1, 1, 2, 40
    softmax_block_size = 256
    grid_size = B * NH * T_q
    shared_mem_size = 2 * softmax_block_size / 32 * 4 #sizeof(float)

    q_ = np.random.rand(B * NH * T_q, T_k).astype(input_type)
    preatt = cp.asarray(q_)
    att = cp.zeros((B * NH * T_q, T_k), dtype=cp.float32)

    softmax_forward_kernel(grid=(grid_size,), block=(softmax_block_size,), 
                            args=(att, preatt, B * NH * T_q, T_k), shared_mem=shared_mem_size)

    inp_tensor = Tensor.from_np(q_).eval()
    out_tensor = Tensor.zeros(q_.shape, input_type).eval()

    out_tensor = device.softmax(out_tensor, inp_tensor)  

    np.testing.assert_allclose(out_tensor.to('cpu').data, cp.asnumpy(att), atol=1e-2, rtol=1e-2)   

def test_transpose_2d():
    x_dim, y_dim = 13, 87
    a = np.random.randn(x_dim, y_dim).astype(np.float32)
    a_t = Tensor.from_np(a).eval()
    out_t = Tensor.zeros((x_dim, y_dim), np.float32).eval()
    a_t.T(out_t)
    out_np = out_t.to('cpu').data
    np.testing.assert_allclose(out_np, a.T, atol=1e-2, rtol=1e-2)

def test_transpose_3d():
    device = Device("cuda")
    shape = (30, 7, 30)
    new_shape = (1, 2, 0)

    q = np.random.randint(low=0, high=10, size=shape).astype(np.float32)
    q_out = Tensor.zeros(shape=shape, dtype=np.float32).eval()
    q_t = Tensor.from_np(q).eval()

    device.transpose(q_out, q_t, axes=new_shape)

    q_np = q_out.to('cpu').data
    q_np.shape = q_out.shape

    np.testing.assert_allclose(q_np, np.transpose(q, axes=new_shape), atol=1e-2, rtol=1e-2)
    
if __name__ == "__main__":
    B, T, OC, NH = 1, 2, 2, 2
    inp = np.random.randint(low =0 , high = 10, size=(B, NH, T, T)).astype(np.float32)
    scaler = np.single([2]).astype(np.float32)
    test_scale_tensor(inp, scaler, B, T, OC, NH)
    test_softmax_v_cp()
    test_softmax(None, None)
    test_transpose_2d()
    test_transpose_3d()
