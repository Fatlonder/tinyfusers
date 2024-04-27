import cudnn
import cupy as cp

handle = cudnn.create_handle()

def conv_2d(X_gpu, W_gpu, padding, stride, dilation):
    graph = cudnn.pygraph(handle = handle, name = "cudnn_graph_conv2d", 
                          io_data_type = cudnn.data_type.HALF, 
                          intermediate_data_type = cudnn.data_type.HALF, 
                          compute_data_type = cudnn.data_type.FLOAT)
    X = graph.tensor_like(X_gpu)
    W = graph.tensor_like(W_gpu)
    Y = graph.conv_fprop(image = X, weight = W, padding = padding, stride = stride, dilation = dilation, compute_data_type = cudnn.data_type.FLOAT)
    Y.set_output(True)
    graph.build([cudnn.heur_mode.A])
    Y_actual = cp.zeros(Y.get_dim(), dtype=cp.float16)
    workspace = cp.empty(graph.get_workspace_size(), dtype=cp.uint8)
    graph.execute({X: X_gpu, W: W_gpu, Y: Y_actual}, workspace, handle= handle)
    return Y_actual