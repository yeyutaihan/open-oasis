import torch
from dit import DiT_models
from vae import VAE_models
from torchvision.io import read_video, write_video
from utils import one_hot_actions, sigmoid_beta_schedule
from tqdm import tqdm
from einops import rearrange
from torch import autocast
device = "cuda:0"

# load DiT checkpoint
ckpt = torch.load("oasis500m.ckpt")
model_prefix = "diffusion_model.model.module."
state_dict = {k.replace(model_prefix, "") : v for k, v in ckpt["state_dict"].items() if k.startswith(model_prefix)}
state_dict = {k : v for k, v in state_dict.items() if "rotary_emb.freqs" not in k}
model = DiT_models["DiT-S/2"]()
model.load_state_dict(state_dict, strict=False)
model = model.to(device).eval()

# load VAE checkpoint
vae_ckpt = torch.load("vit-l-20_ckpt.pt")
vae = VAE_models[vae_ckpt["model_name"]]()
vae.load_state_dict(vae_ckpt["state_dict"])
vae = vae.to(device).eval()

# get input video 
video_id = "Player729-f153ac423f61-20210806-224813.chunk_000"
mp4_path = f"sample_data/{video_id}.mp4"
actions_path = f"sample_data/{video_id}.actions.pt"
video = read_video(mp4_path, pts_unit="sec")[0].float() / 255
actions = one_hot_actions(torch.load(actions_path))
video = video[:10].unsqueeze(0)
actions = actions[:10].unsqueeze(0)

# sampling inputs
n_prompt_frames = 2
x = video[:, :n_prompt_frames]
x = x.to(device)
actions = actions.to(device)

# vae encoding
scaling_factor = 0.07843137255
x = rearrange(x, "b t h w c -> (b t) c h w")
with torch.no_grad():
    x = vae.encode(x * 2 - 1).mean * scaling_factor
x = rearrange(x, "(b t) (h w) c -> b t c h w", t=n_prompt_frames, h=360//20, w=640//20)

# sampling params
batch_size = x.shape[0]
total_frames = 10
max_noise_level = 1000
ddim_noise_steps = 100
stabilization_level = 15
noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1)
noise_abs_max = 20

betas = sigmoid_beta_schedule(max_noise_level).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")

for i in tqdm(range(n_prompt_frames, total_frames)):
    chunk = torch.randn((1, 1, *x.shape[-3:]), device=device)
    chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
    x = torch.cat([x, chunk], dim=1)
    start_frame = max(0, i + 1 - model.max_frames)

    for noise_idx in reversed(range(1, ddim_noise_steps + 1)):
        _t = torch.full((batch_size, i + 1), -1, dtype=torch.long, device=device)
        _t[:, -1] = noise_range[noise_idx]
        _t_next = torch.full((batch_size, i + 1), -1, dtype=torch.long, device=device)
        _t_next[:, -1] = noise_range[noise_idx - 1]
        t = torch.where(_t < 0, stabilization_level - 1, _t)
        t_next = torch.where(_t_next < 0, stabilization_level - 1, _t_next)

        alpha = alphas_cumprod[t]
        mask = rearrange(_t_next < 0, "b t -> b t 1 1 1")
        alpha_next = torch.where(mask, torch.ones_like(alpha), alphas_cumprod[t_next])
        c = (1 - alpha_next).sqrt()
        
        with torch.no_grad():
            with autocast("cuda", dtype=torch.half):
                v = model(x, t, actions[:, start_frame : i + 1])

        x_start = alphas_cumprod[t].sqrt() * x - (1 - alphas_cumprod[t]).sqrt() * v
        x_noise = ((1 / alphas_cumprod[t]).sqrt() * x - x_start) \
                / (1 / alphas_cumprod[t] - 1).sqrt()

        x_pred = alpha_next.sqrt() * x_start + x_noise * c

        mask = rearrange(_t == _t_next, "b t -> b t 1 1 1")
        x_pred = torch.where(mask, x, x_pred.float())
        x = x_pred

# vae decoding
x = rearrange(x, "b t c h w -> (b t) (h w) c")
with torch.no_grad():
    x = (vae.decode(x / scaling_factor) + 1) / 2
x = rearrange(x, "(b t) c h w -> b t h w c", t=total_frames)

x = torch.clamp(x, 0, 1)
x = (x * 255).byte()
write_video("video.mp4", x[0], fps=20)
print("generation saved to video.mp4.")

