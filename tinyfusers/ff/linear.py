import cudnn
import cupy as cp

handle = cudnn.create_handle()

def linear(X_gpu, W_gpu, B_gpu):    
    graph = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT,handle=handle,)
    M, K = X_gpu.shape[1], W_gpu.shape[2]
    X = graph.tensor(name="X", dim=X_gpu.size(), stride=X_gpu.stride(), data_type=X_gpu.dtype,)
    W = graph.tensor(name="W", dim=W_gpu.size(), stride=W_gpu.stride(), data_type=W_gpu.dtype,)
    B = graph.tensor(name="B", dim=B_gpu.size(), stride=B_gpu.stride(), data_type=B_gpu.dtype,)
    response = graph.matmul(name="matmul", A=X, B=W)
    Y = graph.bias(name="bias", input=response, bias=B)
    Y.set_output(True).set_data_type(cudnn.data_type.HALF)
    graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    workspace = cp.empty(graph.get_workspace_size(), dtype=cp.uint8)
    Y_actual = cp.empty((M, K), dtype=cp.float16)
    graph.execute({X: X_gpu, W: W_gpu, B: B_gpu, Y: Y_actual}, workspace, handle=handle)
    return Y_actual

class Linear:
  def __init__(self, in_features, out_features, bias=True):
    self.weight = cp.ones((out_features, in_features), dtype=cp.float32)
    self.bias = cp.ones((out_features), dtype=cp.float32) if bias else None
  def __call__(self, x):
    weight = cp.transpose(self.weight)
    o_tf = cp.dot(x, weight) + self.bias if self.bias is not None else cp.dot(x, weight)
    return o_tf