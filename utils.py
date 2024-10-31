import math
import torch
from torch import nn
from einops import rearrange, parse_shape
from sklearn.decomposition import PCA
from typing import Mapping, Sequence
import torch
from einops import rearrange



def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    f, b = t.shape
    out = a[t]
    return out.reshape(f, b, *((1,) * (len(x_shape) - 2)))


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


ACTION_KEYS = [
    "inventory",
    "ESC",
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
    "forward",
    "back",
    "left",
    "right",
    "cameraX",
    "cameraY",
    "jump",
    "sneak",
    "sprint",
    "swapHands",
    "attack",
    "use",
    "pickItem",
    "drop",
]

def one_hot_actions(actions: Sequence[Mapping[str, int]]) -> torch.Tensor:
    actions_one_hot = torch.zeros(len(actions), len(ACTION_KEYS))
    for i, current_actions in enumerate(actions):
        for j, action_key in enumerate(ACTION_KEYS):
            if action_key.startswith("camera"):
                if action_key == "cameraX":
                    value = current_actions["camera"][0]
                elif action_key == "cameraY":
                    value = current_actions["camera"][1]
                else:
                    raise ValueError(f"Unknown camera action key: {action_key}")
                # NOTE these numbers specific to the camera quantization used in
                # https://github.com/etched-ai/dreamcraft/blob/216e952f795bb3da598639a109bcdba4d2067b69/spark/preprocess_vpt_to_videos_actions.py#L312
                # see method `compress_mouse`
                max_val = 20
                bin_size = 0.5
                num_buckets = int(max_val / bin_size)
                value = (value - num_buckets) / num_buckets
                assert -1 - 1e-3 <= value <= 1 + 1e-3, f"Camera action value must be in [-1, 1], got {value}"
            else:
                value = current_actions[action_key]
                assert 0 <= value <= 1, f"Action value must be in [0, 1] got {value}"
            actions_one_hot[i, j] = value
        
    return actions_one_hot
