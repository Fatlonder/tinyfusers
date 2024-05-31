import cudnn
import math
import cupy as cp
import os
from tinygrad import Tensor

input_type = cp.float32
cuda_dir = os.path.abspath('tinyfusers/native/cuda/')
loaded_from_source = os.path.join(cuda_dir, 'softmax.cu')
def read_code(code_filename):
  with open(code_filename, 'r') as f:
      code = f.read()
      return code
options=('--use_fast_math', '-lcublas -lcublasLt', '-D__CUDA_NO_HALF_CONVERSIONS__', f"-I{cuda_dir}")
module = cp.RawModule(code=read_code(loaded_from_source), backend='nvcc', options=options) 
softmax_forward_kernel = module.get_function('softmax_forward_kernel')

def cudnn_scaled_dot_product_attention(q_gpu, k_gpu, v_gpu):
    b = 4    # batch size
    h = 12   # query number of heads
    s = 1024 # maximum sequence length
    d = 64   # embedding dimension per head
    attn_scale = 1.0 / math.sqrt(d)
    dims = (b, h, s, d)
    strides = (s * h * d, d, h * d, 1)
    graph = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, 
                        intermediate_data_type=cudnn.data_type.FLOAT,
                        compute_data_type=cudnn.data_type.FLOAT,
    )

    q = graph.tensor_like(q_gpu)
    k = graph.tensor_like(k_gpu)
    v = graph.tensor_like(v_gpu)
    o, _ = graph.sdpa(
        name="sdpa",
        q=q, k=k, v=v,
        is_inference=True,
        attn_scale=attn_scale,
        use_causal_mask=True,
    )
    o.set_output(True).set_dim(dims).set_stride(strides)
    graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    o_gpu = cp.empty(o.get_dim(), dtype=cp.float16)
    variant_pack = {
        q: q_gpu,
        k: k_gpu,
        v: v_gpu,
        o: o_gpu,}

    workspace = cp.empty(graph.get_workspace_size(), dtype=cp.uint8)
    graph.execute(variant_pack, workspace)
    return o_gpu

def scaled_dot_product_attention(q_gpu, k_gpu, v_gpu):
    cur_stream = cp.cuda.get_current_stream()
    cur_stream.use()
    q_cp = cp.asarray(q_gpu.numpy())
    k_cp = cp.transpose(cp.asarray(k_gpu.numpy()), axes=(0,1,3,2))
    v_cp = cp.asarray(v_gpu.numpy())
    cur_stream.synchronize()
    cp.cuda.Device().synchronize()
    B, NH, T, HS = q_cp.shape
    softmax_block_size = 256
    grid_size = B * NH * T
    shared_mem_size = 2 * softmax_block_size / 32 * 4 #sizeof(float)
    scale = cp.single(1.0 / math.sqrt(HS))
    att = cp.zeros((B * NH * T, T), dtype=input_type)
    preatt = cp.zeros((B, NH, T, T), dtype=input_type)

    preatt =  scale * cp.matmul(q_cp, k_cp)
    cur_stream.synchronize()
    cp.cuda.Device().synchronize()
    softmax_forward_kernel(grid=(grid_size,), block=(softmax_block_size,), 
                           args=(att, preatt, B * NH * T, T), shared_mem=shared_mem_size)
    att = cp.reshape(att, (B, NH, T, T))
    o = cp.matmul(att, v_cp)
    o_np = cp.asnumpy(o)
    cur_stream.synchronize()
    cp.cuda.Device().synchronize()
    o_tg = Tensor(o_np)
    return o_tg