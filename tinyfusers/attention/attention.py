import cupy as cp
from tinygrad import Tensor
from .sdpa import scaled_dot_product_attention
from ..ff.nn import FeedForward
from ..ff.linear import Linear
from ..ff.group_norm import GroupNorm
from ..ff.layer_norm import LayerNorm
from ..vision.conv2d import Conv2d

class AttnBlock:
  def __init__(self, in_channels):
    self.norm = GroupNorm(32, in_channels)
    self.q = Conv2d(in_channels, in_channels, kernel_size=[1,1])
    self.k = Conv2d(in_channels, in_channels, kernel_size=[1,1])
    self.v = Conv2d(in_channels, in_channels, kernel_size=[1,1])
    self.proj_out = Conv2d(in_channels, in_channels, kernel_size=[1,1])

  # copied from AttnBlock in ldm repo
  def __call__(self, x):
    cur_stream = cp.cuda.get_current_stream()
    cur_stream.use()
    h_ = self.norm(x)
    q,k,v = self.q(h_), self.k(h_), self.v(h_)
    # compute attention. Use cudnn operator when supported.
    b,c,h,w = q.shape
    #h_tf = scaled_dot_product_attention(q,k,v)
    q,k,v = [x.reshape(b,c,h*w).transpose(1,2) for x in (q,k,v)]
    q, k, v = Tensor(cp.asnumpy(q)), Tensor(cp.asnumpy(k)), Tensor(cp.asnumpy(v))
    cur_stream.synchronize()
    cp.cuda.Device().synchronize()
    h_tg = Tensor.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(b,c,h,w)
    #np.testing.assert_allclose(h_tg.numpy(), h_tf.numpy(), atol=1e-2, rtol=1e-2)
    h_tg = cp.asarray(h_tg.numpy())
    cur_stream.synchronize()
    cp.cuda.Device().synchronize()
    o_tg = x + self.proj_out(h_tg)
    return o_tg
  
class CrossAttention:
  def __init__(self, query_dim, context_dim, n_heads, d_head):
    self.to_q = Linear(query_dim, n_heads*d_head, bias=False)
    self.to_k = Linear(context_dim, n_heads*d_head, bias=False)
    self.to_v = Linear(context_dim, n_heads*d_head, bias=False)
    self.num_heads = n_heads
    self.head_size = d_head
    self.to_out = [Linear(n_heads*d_head, query_dim)]

  def __call__(self, x, context=None):
    cur_stream = cp.cuda.get_current_stream()
    cur_stream.use()
    context = x if context is None else context
    q,k,v = self.to_q(x), self.to_k(context), self.to_v(context)
    q, k, v = Tensor(cp.asnumpy(q)), Tensor(cp.asnumpy(k)), Tensor(cp.asnumpy(v))
    cur_stream.synchronize()
    cp.cuda.Device().synchronize()
    q,k,v = [y.reshape(x.shape[0], -1, self.num_heads, self.head_size).transpose(1,2) for y in (q,k,v)]
    attention = Tensor.scaled_dot_product_attention(q, k, v).transpose(1,2)
    h_ = attention.reshape(x.shape[0], -1, self.num_heads * self.head_size)
    o_tf = cp.asarray(h_.numpy())
    cur_stream.synchronize()
    cp.cuda.Device().synchronize()
    o_tf = self.to_out[0](o_tf)
    return o_tf  
  
class BasicTransformerBlock:
  def __init__(self, dim, context_dim, n_heads, d_head):
    self.attn1 = CrossAttention(dim, dim, n_heads, d_head)
    self.ff = FeedForward(dim)
    self.attn2 = CrossAttention(dim, context_dim, n_heads, d_head)
    self.norm1 = LayerNorm(dim)
    self.norm2 = LayerNorm(dim)
    self.norm3 = LayerNorm(dim)

  def __call__(self, x, context=None):
    x = self.attn1(self.norm1(x)) + x
    x = self.attn2(self.norm2(x), context=context) + x
    x = self.ff(self.norm3(x)) + x
    return x

class SpatialTransformer:
  def __init__(self, channels, context_dim, n_heads, d_head):
    self.norm = GroupNorm(32, channels)
    assert channels == n_heads * d_head
    self.proj_in = Conv2d(channels, n_heads * d_head, kernel_size=[1,1])
    self.transformer_blocks = [BasicTransformerBlock(channels, context_dim, n_heads, d_head)]
    self.proj_out = Conv2d(n_heads * d_head, channels, kernel_size=[1,1])

  def __call__(self, x, context=None):
    b, c, h, w = x.shape
    x_in = x
    x = self.norm(x)
    x = self.proj_in(x)
    x = x.reshape(b, c, h*w).permute(0,2,1)
    for block in self.transformer_blocks:
      x = block(x, context=context)
    x = x.permute(0,2,1).reshape(b, c, h, w)
    ret = self.proj_out(x) + x_in
    return ret  

class CLIPAttention:
  def __init__(self):
    self.embed_dim = 768
    self.num_heads = 12
    self.head_dim = self.embed_dim // self.num_heads
    self.k_proj = Linear(self.embed_dim, self.embed_dim)
    self.v_proj = Linear(self.embed_dim, self.embed_dim)
    self.q_proj = Linear(self.embed_dim, self.embed_dim)
    self.out_proj = Linear(self.embed_dim, self.embed_dim)

  def __call__(self, hidden_states, causal_attention_mask):
    cur_stream = cp.cuda.get_current_stream()
    cur_stream.use()
    bsz, tgt_len, embed_dim = hidden_states.shape
    q,k,v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
    q_c, k_c, v_c = Tensor(cp.asnumpy(q)), Tensor(cp.asnumpy(k)), Tensor(cp.asnumpy(v))
    causal_attention_mask = Tensor(cp.asnumpy(causal_attention_mask)).realize()
    cur_stream.synchronize()
    cp.cuda.Device().synchronize()
    q_c,k_c,v_c = [x.reshape(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2) for x in (q_c,k_c,v_c)]
    attn_output = Tensor.scaled_dot_product_attention(q_c, k_c, v_c, attn_mask=causal_attention_mask)
    attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
    attn_output = cp.asarray(attn_output.numpy())
    cur_stream.synchronize()
    cp.cuda.Device().synchronize()
    o_cp = self.out_proj(attn_output)
    return o_cp