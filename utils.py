"""
Adapted from https://github.com/buoyancy99/diffusion-forcing/blob/main/algorithms/diffusion_forcing/models/utils.py
Action format derived from VPT https://github.com/openai/Video-Pre-Training
"""

import math
import torch
from torch import nn
from torchvision.io import read_image, read_video
from torchvision.transforms.functional import resize
from einops import rearrange
from typing import Mapping, Sequence, Tuple, Optional


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


IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
VIDEO_EXTENSIONS = {"mp4"}


def load_prompt(path, video_offset=None, n_prompt_frames=1):
    if path.lower().split(".")[-1] in IMAGE_EXTENSIONS:
        print("prompt is image; ignoring video_offset and n_prompt_frames")
        prompt = read_image(path)
        # add frame dimension
        prompt = rearrange(prompt, "c h w -> 1 c h w")
    elif path.lower().split(".")[-1] in VIDEO_EXTENSIONS:
        prompt = read_video(path, pts_unit="sec")[0]
        if video_offset is not None:
            prompt = prompt[video_offset:]
        prompt = prompt[:n_prompt_frames]
    else:
        raise ValueError(f"unrecognized prompt file extension; expected one in {IMAGE_EXTENSIONS} or {VIDEO_EXTENSIONS}")
    assert prompt.shape[0] == n_prompt_frames, f"input prompt {path} had less than n_prompt_frames={n_prompt_frames} frames"
    prompt = resize(prompt, (360, 640))
    # add batch dimension
    prompt = rearrange(prompt, "t c h w -> 1 t c h w")
    prompt = prompt.float() / 255.0
    return prompt


def load_actions(path, action_offset=None):
    if path.endswith(".actions.pt"):
        actions = one_hot_actions(torch.load(path))
    elif path.endswith(".one_hot_actions.pt"):
        actions = torch.load(path, weights_only=True)
    else:
        raise ValueError("unrecognized action file extension; expected '*.actions.pt' or '*.one_hot_actions.pt'")
    if action_offset is not None:
        actions = actions[action_offset:]
    actions = torch.cat([torch.zeros_like(actions[:1]), actions], dim=0)
    # add batch dimension
    actions = rearrange(actions, "t d -> 1 t d")
    return actions

from colorama import Fore

def cyan(x: str) -> str:
    return f"{Fore.CYAN}{x}{Fore.RESET}"

import wandb

is_rank_zero = wandb.run is not None

import numpy as np
import scipy

class FrechetVideoDistance(nn.Module):
    def __init__(self, path="models/fvd_detector/i3d_torchscript.pt"):
        super().__init__()
        assert path is not None, "FVD detector path must be provided"
        self.detector = torch.jit.load(path).eval()

    def compute_fvd(self, feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
        mu_gen, sigma_gen = self.compute_stats(feats_fake)
        mu_real, sigma_real = self.compute_stats(feats_real)

        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(
            np.dot(sigma_gen, sigma_real), disp=False
        )  # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

        return float(fid)


    def compute_stats(self, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu = feats.mean(axis=0)  # [d]
        sigma = np.cov(feats, rowvar=False)  # [d, d]

        return mu, sigma

    @torch.no_grad()
    def compute(self, videos_fake: torch.Tensor, videos_real: torch.Tensor):
        """
        :param videos_fake: predicted video tensor of shape (frame, batch, channel, height, width)
        :param videos_real: ground-truth observation tensor of shape (frame, batch, channel, height, width)
        :return:
        """
        n_frames, batch_size, c, h, w = videos_fake.shape
        if n_frames < 2:
            raise ValueError("Video must have more than 1 frame for FVD")

        videos_fake = videos_fake.permute(1, 2, 0, 3, 4).contiguous()
        videos_real = videos_real.permute(1, 2, 0, 3, 4).contiguous()

        # detector takes in tensors of shape [batch_size, c, video_len, h, w] with range -1 to 1
        feats_fake = self.detector(videos_fake, **self.detector_kwargs).cpu().numpy()
        feats_real = self.detector(videos_real, **self.detector_kwargs).cpu().numpy()

        return self.compute_fvd(feats_fake, feats_real)

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import (
    structural_similarity_index_measure,
    universal_image_quality_index,
    mean_squared_error,
    peak_signal_noise_ratio
)

def get_validation_metrics_for_videos(
    observation_hat,
    observation_gt,
    lpips_model: Optional[LearnedPerceptualImagePatchSimilarity] = None,
    fid_model: Optional[FrechetInceptionDistance] = None,
    fvd_model: Optional[FrechetVideoDistance] = None,
):
    """
    :param observation_hat: predicted observation tensor of shape (frame, batch, channel, height, width)
    :param observation_gt: ground-truth observation tensor of shape (frame, batch, channel, height, width)
    :param lpips_model: a LearnedPerceptualImagePatchSimilarity object from algorithm.common.metrics
    :param fid_model: a FrechetInceptionDistance object  from algorithm.common.metrics
    :param fvd_model: a FrechetVideoDistance object  from algorithm.common.metrics
    :return: a tuple of metrics
    """
    frame, batch, channel, height, width = observation_hat.shape
    output_dict = {}
    observation_gt = observation_gt.type_as(observation_hat)  # some metrics don't fully support fp16

    if frame < 9:
        fvd_model = None  # FVD requires at least 9 frames

    if fvd_model is not None:
        output_dict["fvd"] = fvd_model.compute(
            torch.clamp(observation_hat, -1.0, 1.0),
            torch.clamp(observation_gt, -1.0, 1.0),
        )

    # reshape to (frame * batch, channel, height, width) for image losses
    observation_hat = observation_hat.view(-1, channel, height, width)
    observation_gt = observation_gt.view(-1, channel, height, width)

    output_dict["mse"] = mean_squared_error(observation_hat, observation_gt)
    output_dict["psnr"] = peak_signal_noise_ratio(observation_hat, observation_gt, data_range=2.0)
    output_dict["ssim"] = structural_similarity_index_measure(observation_hat, observation_gt, data_range=2.0)
    output_dict["uiqi"] = universal_image_quality_index(observation_hat, observation_gt)
    # operations for LPIPS and FID
    observation_hat = torch.clamp(observation_hat, -1.0, 1.0)
    observation_gt = torch.clamp(observation_gt, -1.0, 1.0)

    if lpips_model is not None:
        lpips_model.update(observation_hat, observation_gt)
        lpips = lpips_model.compute().item()
        # Reset the states of non-functional metrics
        output_dict["lpips"] = lpips
        lpips_model.reset()

    if fid_model is not None:
        observation_hat_uint8 = ((observation_hat + 1.0) / 2 * 255).type(torch.uint8)
        observation_gt_uint8 = ((observation_gt + 1.0) / 2 * 255).type(torch.uint8)
        fid_model.update(observation_gt_uint8, real=True)
        fid_model.update(observation_hat_uint8, real=False)
        fid = fid_model.compute()
        output_dict["fid"] = fid
        # Reset the states of non-functional metrics
        fid_model.reset()

    return output_dict


from pathlib import Path
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal, Mapping, Optional, Union
from typing_extensions import override
import os
from wandb_osh.hooks import TriggerWandbSyncHook
import time
from lightning.pytorch.loggers.wandb import WandbLogger, _scan_checkpoints, ModelCheckpoint, Tensor
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.fabric.utilities.types import _PATH


if TYPE_CHECKING:
    from wandb.sdk.lib import RunDisabled
    from wandb.wandb_run import Run


class SpaceEfficientWandbLogger(WandbLogger):
    """
    A wandb logger that by default overrides artifacts to save space, instead of creating new version.
    A variable expiration_days can be set to control how long older versions of artifacts are kept.
    By default, the latest version is kept indefinitely, while older versions are kept for 5 days.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: _PATH = ".",
        version: Optional[str] = None,
        offline: bool = False,
        dir: Optional[_PATH] = None,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        project: Optional[str] = None,
        log_model: Union[Literal["all"], bool] = False,
        experiment: Union["Run", "RunDisabled", None] = None,
        prefix: str = "",
        checkpoint_name: Optional[str] = None,
        expiration_days: Optional[int] = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            save_dir=save_dir,
            version=version,
            offline=False,
            dir=dir,
            id=id,
            anonymous=anonymous,
            project=project,
            log_model=log_model,
            experiment=experiment,
            prefix=prefix,
            checkpoint_name=checkpoint_name,
            **kwargs,
        )

        super().__init__(
            name=name,
            save_dir=save_dir,
            version=version,
            offline=offline,
            dir=dir,
            id=id,
            anonymous=anonymous,
            project=project,
            log_model=log_model,
            experiment=experiment,
            prefix=prefix,
            checkpoint_name=checkpoint_name,
            **kwargs,
        )
        self.expiration_days = expiration_days
        self._last_artifacts = []

    def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        import wandb

        # get checkpoints to be saved with associated score
        checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)

        # log iteratively all new checkpoints
        artifacts = []
        for t, p, s, tag in checkpoints:
            metadata = {
                "score": s.item() if isinstance(s, Tensor) else s,
                "original_filename": Path(p).name,
                checkpoint_callback.__class__.__name__: {
                    k: getattr(checkpoint_callback, k)
                    for k in [
                        "monitor",
                        "mode",
                        "save_last",
                        "save_top_k",
                        "save_weights_only",
                        "_every_n_train_steps",
                    ]
                    # ensure it does not break if `ModelCheckpoint` args change
                    if hasattr(checkpoint_callback, k)
                },
            }
            if not self._checkpoint_name:
                self._checkpoint_name = f"model-{self.experiment.id}"

            artifact = wandb.Artifact(name=self._checkpoint_name, type="model", metadata=metadata)
            artifact.add_file(p, name="model.ckpt")
            aliases = ["latest", "best"] if p == checkpoint_callback.best_model_path else ["latest"]
            self.experiment.log_artifact(artifact, aliases=aliases)
            # remember logged models - timestamp needed in case filename didn't change (lastkckpt or custom name)
            self._logged_model_time[p] = t
            artifacts.append(artifact)

        for artifact in self._last_artifacts:
            if not self._offline:
                artifact.wait()
            artifact.ttl = timedelta(days=self.expiration_days)
            artifact.save()
        self._last_artifacts = artifacts


class OfflineWandbLogger(SpaceEfficientWandbLogger):
    """
    Wraps WandbLogger to trigger offline sync hook occasionally.
    This is useful when running on slurm clusters, many of which
    only has internet on login nodes, not compute nodes.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: _PATH = ".",
        version: Optional[str] = None,
        offline: bool = False,
        dir: Optional[_PATH] = None,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        project: Optional[str] = None,
        log_model: Union[Literal["all"], bool] = False,
        experiment: Union["Run", "RunDisabled", None] = None,
        prefix: str = "",
        checkpoint_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            save_dir=save_dir,
            version=version,
            offline=False,
            dir=dir,
            id=id,
            anonymous=anonymous,
            project=project,
            log_model=log_model,
            experiment=experiment,
            prefix=prefix,
            checkpoint_name=checkpoint_name,
            **kwargs,
        )
        self._offline = offline
        communication_dir = Path(".wandb_osh_command_dir")
        communication_dir.mkdir(parents=True, exist_ok=True)
        self.trigger_sync = TriggerWandbSyncHook(communication_dir)
        self.last_sync_time = 0.0
        self.min_sync_interval = 60
        self.wandb_dir = os.path.join(self._save_dir, "wandb/latest-run")

    @override
    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        out = super().log_metrics(metrics, step)
        if time.time() - self.last_sync_time > self.min_sync_interval:
            self.trigger_sync(self.wandb_dir)
            self.last_sync_time = time.time()
        return out

def is_run_id(run_id: str) -> bool:
    """Check if a string is a run ID."""
    return len(run_id) == 8 and run_id.isalnum()


def version_to_int(artifact) -> int:
    """Convert versions of the form vX to X. For example, v12 to 12."""
    return int(artifact.version[1:])

def download_latest_checkpoint(run_path: str, download_dir: Path) -> Path:
    api = wandb.Api()
    run = api.run(run_path)

    # Find the latest saved model checkpoint.
    latest = None
    for artifact in run.logged_artifacts():
        if artifact.type != "model" or artifact.state != "COMMITTED":
            continue

        if latest is None or version_to_int(artifact) > version_to_int(latest):
            latest = artifact

    # Download the checkpoint.
    download_dir.mkdir(exist_ok=True, parents=True)
    root = download_dir / run_path
    latest.download(root=root)
    return root / "model.ckpt"

def log_video(
    observation_hat,
    observation_gt=None,
    step=0,
    namespace="train",
    prefix="video",
    context_frames=0,
    color=(255, 0, 0),
    logger=None,
):
    """
    take in video tensors in range [-1, 1] and log into wandb

    :param observation_hat: predicted observation tensor of shape (frame, batch, channel, height, width)
    :param observation_gt: ground-truth observation tensor of shape (frame, batch, channel, height, width)
    :param step: an int indicating the step number
    :param namespace: a string specify a name space this video logging falls under, e.g. train, val
    :param prefix: a string specify a prefix for the video name
    :param context_frames: an int indicating how many frames in observation_hat are ground truth given as context
    :param color: a tuple of 3 numbers specifying the color of the border for ground truth frames
    :param logger: optional logger to use. use global wandb if not specified
    """
    if not logger:
        logger = wandb
    if observation_gt is None:
        observation_gt = torch.zeros_like(observation_hat)
    observation_hat[:context_frames] = observation_gt[:context_frames]
    # Add red border of 1 pixel width to the context frames
    for i, c in enumerate(color):
        c = c / 255.0
        observation_hat[:context_frames, :, i, [0, -1], :] = c
        observation_hat[:context_frames, :, i, :, [0, -1]] = c
        observation_gt[:, :, i, [0, -1], :] = c
        observation_gt[:, :, i, :, [0, -1]] = c
    video = torch.cat([observation_hat, observation_gt], -1).detach().cpu().numpy()
    video = np.transpose(np.clip(video, a_min=0.0, a_max=1.0) * 255, (1, 0, 2, 3, 4)).astype(np.uint8)
    # video[..., 1:] = video[..., :1]  # remove framestack, only visualize current frame
    n_samples = len(video)
    # use wandb directly here since pytorch lightning doesn't support logging videos yet
    for i in range(n_samples):
        logger.log(
            {
                f"{namespace}/step_{step}/{i}": wandb.Video(video[i], fps=24),
                f"trainer/global_step": step,
            }
        )

def extract(a, t, x_shape):
    f, b = t.shape
    out = a[t]
    return out.reshape(f, b, *((1,) * (len(x_shape) - 2)))