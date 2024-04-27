# Diffusion based models built around tinygrad (for now).
The code for stable diffusion is mainly copied from tinygrad's sd example.  
Compared to diffusers library this example takes lot more. 
The goal is to make tinyfusers faster but also usable.  

You need to install `nvidia-cudnn-cu12` and `cudnn-frontend` since kernels rely on them.  
Run the sd example using: `python3 -m  example.sd1`