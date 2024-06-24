import cudnn
import math
import cupy as cp
import os

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

def scaled_dot_product_attention(q_cp, k_cp, v_cp, attn_mask=None):
    cur_stream = cp.cuda.get_current_stream()
    cur_stream.use()
    B, NH, T_q, HS = q_cp.shape
    T_k, T_v, HS_V = k_cp.shape[-2], v_cp.shape[-2], v_cp.shape[-1]
    k_cp = cp.transpose(k_cp, axes=(0,1,3,2))
    softmax_block_size = 256
    grid_size = B * NH * T_q
    shared_mem_size = 2 * softmax_block_size / 32 * 4 #sizeof(float)
    scale = cp.single(1.0 / math.sqrt(HS))
    preatt = cp.zeros((B, NH, T_q, T_k), dtype=input_type)
    att = cp.zeros((B * NH * T_q, T_k), dtype=input_type)

    preatt =  scale * cp.matmul(q_cp, k_cp)
    if attn_mask is not None:
        preatt = preatt + cp.where(attn_mask == 0, -float("inf"), 0) if attn_mask.dtype == cp.bool_ else preatt + attn_mask

    cur_stream.synchronize()
    cp.cuda.Device().synchronize()

    softmax_forward_kernel(grid=(grid_size,), block=(softmax_block_size,), 
                           args=(att, preatt, B * NH * T_q, T_k), shared_mem=shared_mem_size)
    att = cp.reshape(att, (B, NH, T_q, T_k))
    o_tg = cp.matmul(att, v_cp)
    return o_tg