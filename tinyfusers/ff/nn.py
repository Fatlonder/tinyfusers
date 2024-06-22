import cupy as cp
from tinygrad import Tensor
from .linear import  Linear

class GEGLU:
  def __init__(self, dim_in, dim_out):
    self.proj = Linear(dim_in, dim_out * 2)
    self.dim_out = dim_out

  def __call__(self, x):
    x, gate = self.proj(x).chunk(2, dim=-1)
    return x * gate.gelu()

class FeedForward:
  def __init__(self, dim, mult=4):
    self.net = [
      GEGLU(dim, dim*mult),
      lambda x: x,  # needed for weights loading code to work
      Linear(dim*mult, dim)
    ]

  def __call__(self, x):
    return x.sequential(self.net)

class CLIPMLP:
  def __init__(self):
    self.fc1 = Linear(768, 3072)
    self.fc2 = Linear(3072, 768)

  def __call__(self, hidden_states):
    cur_stream = cp.cuda.get_current_stream()
    cur_stream.use()
    hidden_states = self.fc1(hidden_states)
    ot = Tensor(cp.asnumpy(hidden_states))
    cur_stream.synchronize()
    cp.cuda.Device().synchronize()
    hidden_states = ot.quick_gelu()
    hidden_states = cp.asarray(hidden_states.numpy())
    cur_stream.synchronize()
    cp.cuda.Device().synchronize()
    hidden_states = self.fc2(hidden_states)
    return hidden_states