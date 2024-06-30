import sys
sys.path.insert(0, 'tinyfusers')

import tinyfusers
import importlib
importlib.reload(tinyfusers)
print(tinyfusers.__file__)

import argparse
from IPython.display import display
from PIL import Image
from tqdm import tqdm
import numpy as np
import cupy as cp
from tinygrad import GlobalCounters, Tensor
from tinygrad.nn.state import torch_load
from tinygrad.helpers import Timing, Context, getenv, fetch
from tinyfusers.variants.sd import StableDiffusion
from tinyfusers.tokenizer.clip import ClipTokenizer
from tinyfusers.storage.state import update_state
from tinyfusers.storage.unpicker import load_weights

if __name__ == "__main__":
  default_prompt = "a horse sized cat eating a bagel"
  parser = argparse.ArgumentParser(description='Run Stable Diffusion', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--steps', type=int, default=5, help="Number of steps in diffusion")
  parser.add_argument('--prompt', type=str, default=default_prompt, help="Phrase to render")
  parser.add_argument('--noshow', action='store_true', help="Don't show the image")
  parser.add_argument('--fp16', action='store_true', help="Cast the weights to float16")
  parser.add_argument('--timing', action='store_true', help="Print timing per step")
  parser.add_argument('--seed', type=int, help="Set the random latent seed")
  parser.add_argument('--guidance', type=float, default=7.5, help="Prompt strength")
  args = parser.parse_args()

  model = StableDiffusion()

  # load in weights
  default_prompt = "a horse sized cat eating a bagel"
  args = {"prompt": default_prompt, "steps": 20, "fp16": True, "out": "rendered.png", "noshow": False, "timing": False, "guidance":7.5, "seed": 42}
  state_dictionary = torch_load(fetch('https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt', 'sd-v1-4.ckpt'))
  update_state(model, state_dictionary['state_dict'])

  # run through CLIP to get context
  tokenizer = ClipTokenizer()
  prompt = cp.array([tokenizer.encode(args['prompt'])])
  context = model.cond_stage_model.transformer.text_model(prompt)

  prompt = cp.array([tokenizer.encode("")])
  unconditional_context = model.cond_stage_model.transformer.text_model(prompt)

  print(f"CLIP context: {context.shape}, unconditional CLIP context: {unconditional_context.shape}")
  del model.cond_stage_model

  timesteps = list(range(1, 1000, 1000//args['steps']))
  print(f"running for {timesteps} timesteps")
  alphas = model.alphas_cumprod[timesteps]
  alphas_prev = cp.concatenate((cp.array([1.0]), alphas[:-1])).astype(cp.float32)

  cur_stream = cp.cuda.get_current_stream()
  cur_stream.use()
  if args['seed'] is not None: Tensor._seed = args['seed']
  latent = Tensor.randn(1,4,64,64)
  latent = cp.asarray(latent.numpy())
  cur_stream.synchronize()
  cp.cuda.Device().synchronize()

  with Context(BEAM=getenv("LATEBEAM")):
    for index, timestep in (t:=tqdm(list(enumerate(timesteps))[::-1])):
      GlobalCounters.reset()
      t.set_description("%3d %3d" % (index, timestep))
      with Timing("step in ", enabled=args['timing'], on_exit=lambda _: f", using {GlobalCounters.mem_used/1e9:.2f} GB"):
        tid = cp.array([index])
        latent = model(unconditional_context, context, latent, cp.array([timestep]), alphas[tid], alphas_prev[tid], cp.array([args['guidance']]))
  x = model.decode(latent)
  print(x.shape)
  im = Image.fromarray(cp.asnumpy(x).astype(np.uint8, copy=False))
  print(f"saving {args['out']}")
  im.save(args['out'])
  display(im)