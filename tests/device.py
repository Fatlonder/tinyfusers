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

if __name__ == "__main__":
    B, T, OC, NH = 1, 2, 2, 2
    inp = np.random.randint(low =0 , high = 10, size=(B, NH, T, T)).astype(np.float32)
    scaler = np.single([2]).astype(np.float32)
    test_scale_tensor(inp, scaler, B, T, OC, NH)
