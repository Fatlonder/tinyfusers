from .encoder import Encoder
from .decoder import Decoder
from ..vision.conv2d import Conv2d


class AutoencoderKL:
  def __init__(self):
    self.encoder = Encoder()
    self.decoder = Decoder()
    self.quant_conv = Conv2d(8, 8, kernel_size=[1,1])
    self.post_quant_conv = Conv2d(4, 4, kernel_size=[1,1])

  def __call__(self, x):
    latent = self.encoder(x)
    latent = self.quant_conv(latent)
    latent = latent[:, 0:4]  # only the means
    print("latent", latent.shape)
    latent = self.post_quant_conv(latent)
    return self.decoder(latent)