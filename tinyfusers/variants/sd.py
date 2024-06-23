from collections import namedtuple
import cupy as cp
from ..vision.unet import UNetModel
from ..vae.vae import AutoencoderKL
from ..vae.encoder import CLIPTextTransformer

class StableDiffusion:
  def __init__(self):
    self.alphas_cumprod = get_alphas_cumprod()
    self.model = namedtuple("DiffusionModel", ["diffusion_model"])(diffusion_model = UNetModel())
    self.first_stage_model = AutoencoderKL()
    self.cond_stage_model = namedtuple("CondStageModel", ["transformer"])(transformer = namedtuple("Transformer", ["text_model"])(text_model = CLIPTextTransformer()))

  def get_x_prev_and_pred_x0(self, x, e_t, a_t, a_prev):
    temperature = 1
    sigma_t = 0
    sqrt_one_minus_at = cp.sqrt(1-a_t)
    #print(a_t, a_prev, sigma_t, sqrt_one_minus_at)

    pred_x0 = (x - sqrt_one_minus_at * e_t) / cp.sqrt(a_t)

    # direction pointing to x_t
    dir_xt = cp.sqrt(1. - a_prev - sigma_t**2) * e_t

    x_prev = cp.sqrt(a_prev) * pred_x0 + dir_xt
    return x_prev, pred_x0

  def get_model_output(self, unconditional_context, context, latent, timestep, unconditional_guidance_scale):
    # put into diffuser
    ltnt_brdcst = cp.broadcast_to(latent, (2, *latent.shape[1:]))
    unc_cntx = cp.concatenate((unconditional_context, context), axis=0) 
    latents = self.model.diffusion_model(ltnt_brdcst, timestep, unc_cntx)
    unconditional_latent, latent = latents[0:1], latents[1:2]

    e_t = unconditional_latent + unconditional_guidance_scale * (latent - unconditional_latent)
    return e_t

  def decode(self, x):
    x = self.first_stage_model.post_quant_conv(1/0.18215 * x)
    x = self.first_stage_model.decoder(x)

    # make image correct size and scale
    x = (x + 1.0) / 2.0
    x = cp.clip(cp.transpose(x.reshape(3,512,512), (1,2,0)), 0,1) * 255
    return x.astype(cp.uint8)

  def __call__(self, unconditional_context, context, latent, timestep, alphas, alphas_prev, guidance):
    e_t = self.get_model_output(unconditional_context, context, latent, timestep, guidance)
    x_prev, _ = self.get_x_prev_and_pred_x0(latent, e_t, alphas, alphas_prev)
    #e_t_next = get_model_output(x_prev)
    #e_t_prime = (e_t + e_t_next) / 2
    #x_prev, pred_x0 = get_x_prev_and_pred_x0(latent, e_t_prime, index)
    return x_prev

def get_alphas_cumprod(beta_start=0.00085, beta_end=0.0120, n_training_steps=1000):
  betas = cp.linspace(beta_start ** 0.5, beta_end ** 0.5, n_training_steps, dtype=cp.float32) ** 2
  alphas = 1.0 - betas
  alphas_cumprod = cp.cumprod(alphas, axis=0)
  return alphas_cumprod