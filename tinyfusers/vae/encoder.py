import cupy as cp
from .mid import Mid
from ..attention.attention import CLIPAttention
from ..vision.resnet import ResnetBlock
from ..vision.conv2d import Conv2d
from ..ff.nn import CLIPMLP
from ..ff.layer_norm import LayerNorm
from ..ff.group_norm import GroupNorm
from ..ff.embedding import Embedding
from ..tensor.tensor import Tensor

class Encoder:
  def __init__(self):
    sz = [(128, 128), (128, 256), (256, 512), (512, 512)]
    self.conv_in = Conv2d(3, 128, kernel_size=[3,3], padding=[1,1])
    arr = []
    for i,s in enumerate(sz):
      arr.append({"block":[ResnetBlock(s[0], s[1]), ResnetBlock(s[1], s[1])]})
      if i != 3: arr[-1]['downsample'] = {"conv": Conv2d(s[1], s[1], kernel_size=[3,3], stride=[2,2], padding=[0,1,0,1])}
    self.down = arr

    self.mid = Mid(512)
    self.norm_out = GroupNorm(32, 512)
    self.conv_out = Conv2d(512, 8, kernel_size=[3,3], padding=[1,1])

  def __call__(self, x):
    x = self.conv_in(x)
    for l in self.down:
      print("encode", x.shape)
      for b in l['block']: x = b(x)
      if 'downsample' in l: x = l['downsample']['conv'](x)
    x = self.mid(x)
    out = self.conv_out(Tensor.swish(self.norm_out(x)))
    return out

class CLIPEncoder:
  def __init__(self):
    self.layers = [CLIPEncoderLayer() for i in range(12)]

  def __call__(self, hidden_states, causal_attention_mask):
    for l in self.layers:
      hidden_states = l(hidden_states, causal_attention_mask)    
    return hidden_states

class CLIPEncoderLayer:
  def __init__(self):
    self.self_attn = CLIPAttention()
    self.layer_norm1 = LayerNorm(768)
    self.mlp = CLIPMLP()
    self.layer_norm2 = LayerNorm(768)

  def __call__(self, hidden_states, causal_attention_mask):
    residual = hidden_states
    hidden_states = self.layer_norm1(hidden_states)
    hidden_states = self.self_attn(hidden_states, causal_attention_mask)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states
  
class CLIPTextEmbeddings:
  def __init__(self):
    self.token_embedding = Embedding(49408, 768)
    self.position_embedding = Embedding(77, 768)
  def __call__(self, input_ids, position_ids):
    return self.token_embedding(input_ids) + self.position_embedding(position_ids)

class CLIPTextTransformer:
  def __init__(self):
    self.embeddings = CLIPTextEmbeddings()
    self.encoder = CLIPEncoder()
    self.final_layer_norm = LayerNorm(768)

  def __call__(self, input_ids):
    x = self.embeddings(input_ids, cp.arange(input_ids.shape[1]).reshape(1, -1).astype(cp.float32))
    x = self.encoder(x, cp.triu(cp.full((1, 1, 77, 77), float("-inf")), k=1).astype(cp.float32))
    out = self.final_layer_norm(x)
    return out