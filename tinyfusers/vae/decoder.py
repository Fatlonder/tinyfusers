from ..vision.resnet import ResnetBlock
from .mid import Mid
from tinygrad.nn import GroupNorm
from ..vision.conv2d4 import Conv2d


class Decoder:
  def __init__(self):
    sz = [(128, 256), (256, 512), (512, 512), (512, 512)]
    self.conv_in = Conv2d(4,512, kernel_size=[3,3], padding=[1,1])
    self.mid = Mid(512)

    arr = []
    for i,s in enumerate(sz):
      arr.append({"block":
        [ResnetBlock(s[1], s[0]),
         ResnetBlock(s[0], s[0]),
         ResnetBlock(s[0], s[0])]})
      if i != 0: arr[-1]['upsample'] = {"conv": Conv2d(s[0], s[0], kernel_size=[3,3], padding=[1,1])}
    self.up = arr

    self.norm_out = GroupNorm(32, 128)
    self.conv_out = Conv2d(128, 3, kernel_size=[3,3], padding=[1,1])

  def __call__(self, x):
    x = self.conv_in(x)
    x = self.mid(x)

    for l in self.up[::-1]:
      print("decode", x.shape)
      for b in l['block']: x = b(x)
      if 'upsample' in l:
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html ?
        bs,c,py,px = x.shape
        x = x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, 2, px, 2).reshape(bs, c, py*2, px*2)
        x = l['upsample']['conv'](x)
      x.realize()

    return self.conv_out(self.norm_out(x).swish())