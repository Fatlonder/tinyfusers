import cudnn
import math
import cupy as cp

def scaled_dot_product_attention(q_gpu, k_gpu, v_gpu):
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