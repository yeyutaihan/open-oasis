"""
References:
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing
"""
import torch
from dit import DiT_models
from vae import VAE_models
from torchvision.io import read_video, write_video
from utils import one_hot_actions, sigmoid_beta_schedule
from tqdm import tqdm
from einops import rearrange
from torch import autocast
from safetensors.torch import load_model
import argparse
assert torch.cuda.is_available()
device = "cuda:0"

parse = argparse.ArgumentParser()

parse.add_argument('--oasis-ckpt', type=str, help='Path to Oasis DiT checkpoint.', default="oasis500m.safetensors")
parse.add_argument('--vae-ckpt', type=str, help='Path to Oasis ViT-VAE checkpoint.', default="vit-l-20.safetensors")
parse.add_argument('--num-frames', type=int, help='How many frames should be generated?', default=32)
parse.add_argument('--output-path', type=str, help='Path where generated video should be saved.', default="video.mp4")
parse.add_argument('--fps', type=int, help='What framerate should be used to save the output?', default=20)
parse.add_argument('--ddim-steps', type=int, help='How many DDIM steps?', default=50)

args = parse.parse_args()

# load DiT checkpoint
model = DiT_models["DiT-S/2"]()
if args.oasis_ckpt.endswith(".pt"):
    ckpt = torch.load(args.oasis_ckpt, weights_only=True)
    model.load_state_dict(ckpt, strict=False)
elif args.oasis_ckpt.endswith(".safetensors"):
    load_model(model, args.oasis_ckpt)
model = model.to(device).eval()

# load VAE checkpoint
vae = VAE_models["vit-l-20-shallow-encoder"]()
if args.vae_ckpt.endswith(".pt"):
    vae_ckpt = torch.load(args.vae_ckpt, weights_only=True)
    vae.load_state_dict(vae_ckpt)
elif args.vae_ckpt.endswith(".safetensors"):
    load_model(vae, args.vae_ckpt)
vae = vae.to(device).eval()

# sampling params
B = 1
total_frames = args.num_frames
max_noise_level = 1000
ddim_noise_steps = args.ddim_steps
noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1)
noise_abs_max = 20
ctx_max_noise_idx = ddim_noise_steps // 10 * 3

# get input video 
video_id = "snippy-chartreuse-mastiff-f79998db196d-20220401-224517.chunk_001"
mp4_path = f"sample_data/{video_id}.mp4"
actions_path = f"sample_data/{video_id}.actions.pt"
video = read_video(mp4_path, pts_unit="sec")[0].float() / 255
actions = one_hot_actions(torch.load(actions_path))
offset = 100
video = video[offset:offset+total_frames].unsqueeze(0)
actions = actions[offset:offset+total_frames].unsqueeze(0)

# sampling inputs
n_prompt_frames = 1
x = video[:, :n_prompt_frames]
x = x.to(device)
actions = actions.to(device)

# vae encoding
scaling_factor = 0.07843137255
x = rearrange(x, "b t h w c -> (b t) c h w")
H, W = x.shape[-2:]
with torch.no_grad():
    x = vae.encode(x * 2 - 1).mean * scaling_factor
x = rearrange(x, "(b t) (h w) c -> b t c h w", t=n_prompt_frames, h=H//vae.patch_size, w=W//vae.patch_size)

# get alphas
betas = sigmoid_beta_schedule(max_noise_level).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")

# sampling loop
for i in tqdm(range(n_prompt_frames, total_frames)):
    chunk = torch.randn((B, 1, *x.shape[-3:]), device=device)
    chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
    x = torch.cat([x, chunk], dim=1)
    start_frame = max(0, i + 1 - model.max_frames)

    for noise_idx in reversed(range(1, ddim_noise_steps + 1)):
        # set up noise values
        ctx_noise_idx = min(noise_idx, ctx_max_noise_idx)
        t_ctx  = torch.full((B, i), noise_range[ctx_noise_idx], dtype=torch.long, device=device)
        t      = torch.full((B, 1), noise_range[noise_idx],     dtype=torch.long, device=device)
        t_next = torch.full((B, 1), noise_range[noise_idx - 1], dtype=torch.long, device=device)
        t_next = torch.where(t_next < 0, t, t_next)
        t = torch.cat([t_ctx, t], dim=1)
        t_next = torch.cat([t_ctx, t_next], dim=1)

        # sliding window
        x_curr = x.clone()
        x_curr = x_curr[:, start_frame:]
        t = t[:, start_frame:]
        t_next = t_next[:, start_frame:]

        # add some noise to the context
        ctx_noise = torch.randn_like(x_curr[:, :-1])
        ctx_noise = torch.clamp(ctx_noise, -noise_abs_max, +noise_abs_max)
        x_curr[:, :-1] = alphas_cumprod[t[:, :-1]].sqrt() * x_curr[:, :-1] + (1 - alphas_cumprod[t[:, :-1]]).sqrt() * ctx_noise

        # get model predictions
        with torch.no_grad():
            with autocast("cuda", dtype=torch.half):
                v = model(x_curr, t, actions[:, start_frame : i + 1])

        x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
        x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) \
                / (1 / alphas_cumprod[t] - 1).sqrt()

        # get frame prediction
        x_pred = alphas_cumprod[t_next].sqrt() * x_start + x_noise * (1 - alphas_cumprod[t_next]).sqrt()
        x[:, -1:] = x_pred[:, -1:]

# vae decoding
x = rearrange(x, "b t c h w -> (b t) (h w) c")
with torch.no_grad():
    x = (vae.decode(x / scaling_factor) + 1) / 2
x = rearrange(x, "(b t) c h w -> b t h w c", t=total_frames)

# save video
x = torch.clamp(x, 0, 1)
x = (x * 255).byte()
write_video(args.output_path, x[0].cpu(), fps=args.fps)
print(f"generation saved to {args.output_path}.")

