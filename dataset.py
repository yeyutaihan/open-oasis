from typing import Sequence
import torch
import random
import os
import numpy as np
from omegaconf import DictConfig
from torchvision import transforms
from pathlib import Path
import json
from pytorchvideo.data.encoded_video import EncodedVideo

class MinecraftVideoDataset(torch.utils.data.Dataset):
    """
    Minecraft dataset
    """

    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.resolution = cfg.resolution
        self.external_cond_dim = cfg.external_cond_dim
        self.n_frames = (
            cfg.n_frames * cfg.frame_skip
            if split == "training"
            else cfg.n_frames * cfg.frame_skip * cfg.validation_multiplier
        )
        self.frame_skip = cfg.frame_skip
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.split_dir = self.save_dir / f"{split}"

        self.metadata_path = self.save_dir / "metadata.json"

        if not self.metadata_path.exists():
            # Build dataset
            print(f"Creating dataset in {self.save_dir}...")
            json.dump(
                {
                    "training": self.get_data_lengths("training"),
                    "validation": self.get_data_lengths("validation"),
                },
                open(self.metadata_path, "w"),
            )

        self.metadata = json.load(open(self.metadata_path, "r"))
        self.data_paths = self.get_data_paths(self.split)
        self.clips_per_video = np.clip(np.array(self.metadata[split]) - self.n_frames + 1, a_min=1, a_max=None).astype(
            np.int32
        )
        self.cum_clips_per_video = np.cumsum(self.clips_per_video)
        self.transform = transforms.Resize((self.resolution, self.resolution), antialias=True)

        # shuffle but keep the same order for each epoch, so validation sample is diverse yet deterministic
        random.seed(0)
        self.idx_remap = list(range(self.__len__()))
        random.shuffle(self.idx_remap)

    def __len__(self):
        return self.clips_per_video.sum()

    def get_data_paths(self, split):
        data_dir = self.save_dir / split
        paths = list(data_dir.glob("*.npz"))
        return paths

    def get_data_lengths(self, split):
        lengths = [300] * len(self.get_data_paths(split))
        return lengths

    def split_idx(self, idx):
        video_idx = np.argmax(self.cum_clips_per_video > idx)
        frame_idx = idx - np.pad(self.cum_clips_per_video, (1, 0))[video_idx]
        return video_idx, frame_idx

    def __getitem__(self, idx):
        idx = self.idx_remap[idx]
        file_idx, frame_idx = self.split_idx(idx)
        action_path = self.data_paths[file_idx]
        video_path = action_path.with_suffix(".mp4")
        video = EncodedVideo.from_path(video_path, decode_audio=False)
        video = video.get_clip(start_sec=0.0, end_sec=video.duration)["video"]
        video = video.permute(1, 2, 3, 0).numpy()
        actions = np.load(action_path)["actions"][1:]

        video = video[frame_idx : frame_idx + self.n_frames]  # (t, h, w, 3)
        actions = actions[frame_idx : frame_idx + self.n_frames]  # (t, )
        actions = np.eye(4)[actions]  # (t, 4)

        pad_len = self.n_frames - len(video)

        nonterminal = np.ones(self.n_frames)
        if len(video) < self.n_frames:
            video = np.pad(video, ((0, pad_len), (0, 0), (0, 0), (0, 0)))
            actions = np.pad(actions, ((0, pad_len),))
            nonterminal[-pad_len:] = 0

        video = torch.from_numpy(video / 255.0).float().permute(0, 3, 1, 2).contiguous()
        video = self.transform(video)

        return (
            video[:: self.frame_skip],
            torch.tensor(actions[:: self.frame_skip], dtype=torch.float32),
            nonterminal[:: self.frame_skip],
        )