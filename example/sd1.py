import argparse
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np
import tempfile
from tinygrad import Device, GlobalCounters, dtypes, Tensor, TinyJit
from tinygrad.nn.state import torch_load, load_state_dict, get_state_dict
from tinygrad.helpers import Timing, Context, getenv, fetch, colored
from tinyfusers.variants.sd import StableDiffusion
from tinyfusers.tokenizer.clip import ClipTokenizer

if __name__ == "__main__":
  default_prompt = "a horse sized cat eating a bagel"
  parser = argparse.ArgumentParser(description='Run Stable Diffusion', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--steps', type=int, default=5, help="Number of steps in diffusion")
  parser.add_argument('--prompt', type=str, default=default_prompt, help="Phrase to render")
  parser.add_argument('--out', type=str, default=Path(tempfile.gettempdir()) / "rendered.png", help="Output filename")
  parser.add_argument('--noshow', action='store_true', help="Don't show the image")
  parser.add_argument('--fp16', action='store_true', help="Cast the weights to float16")
  parser.add_argument('--timing', action='store_true', help="Print timing per step")
  parser.add_argument('--seed', type=int, help="Set the random latent seed")
  parser.add_argument('--guidance', type=float, default=7.5, help="Prompt strength")
  args = parser.parse_args()

  Tensor.no_grad = True
  model = StableDiffusion()

  # load in weights
  load_state_dict(model, torch_load(fetch('https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt', 'sd-v1-4.ckpt'))['state_dict'], strict=False)

  if args.fp16:
    for l in get_state_dict(model).values():
      l.replace(l.cast(dtypes.float16).realize())

  # run through CLIP to get context
  tokenizer = ClipTokenizer()
  prompt = Tensor([tokenizer.encode(args.prompt)])
  context = model.cond_stage_model.transformer.text_model(prompt).realize()
  print("got CLIP context", context.shape)

  prompt = Tensor([tokenizer.encode("")])
  unconditional_context = model.cond_stage_model.transformer.text_model(prompt).realize()
  print("got unconditional CLIP context", unconditional_context.shape)

  # done with clip model
  del model.cond_stage_model

  timesteps = list(range(1, 1000, 1000//args.steps))
  print(f"running for {timesteps} timesteps")
  alphas = model.alphas_cumprod[Tensor(timesteps)]
  alphas_prev = Tensor([1.0]).cat(alphas[:-1])

  # start with random noise
  if args.seed is not None: Tensor._seed = args.seed
  latent = Tensor.randn(1,4,64,64)

  @TinyJit
  def run(model, *x): return model(*x).realize()

  # this is diffusion
  with Context(BEAM=getenv("LATEBEAM")):
    for index, timestep in (t:=tqdm(list(enumerate(timesteps))[::-1])):
      GlobalCounters.reset()
      t.set_description("%3d %3d" % (index, timestep))
      with Timing("step in ", enabled=args.timing, on_exit=lambda _: f", using {GlobalCounters.mem_used/1e9:.2f} GB"):
        tid = Tensor([index])
        latent = run(model, unconditional_context, context, latent, Tensor([timestep]), alphas[tid], alphas_prev[tid], Tensor([args.guidance]))
        if args.timing: Device[Device.DEFAULT].synchronize()
    del run

  # upsample latent space to image with autoencoder
  x = model.decode(latent)
  print(x.shape)

  # save image
  im = Image.fromarray(x.numpy().astype(np.uint8, copy=False))
  print(f"saving {args.out}")
  im.save(args.out)
  # Open image.
  if not args.noshow: im.show()

  # validation!
  if args.prompt == default_prompt and args.steps == 5 and args.seed == 0 and args.guidance == 7.5:
    ref_image = Tensor(np.array(Image.open(Path(__file__).parent / "stable_diffusion_seed0.png")))
    distance = (((x - ref_image).cast(dtypes.float) / ref_image.max())**2).mean().item()
    assert distance < 3e-4, f"validation failed with {distance=}"
    print(colored(f"output validated with {distance=}", "green"))