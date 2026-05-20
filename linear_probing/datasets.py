import os

import numpy as np
import pandas as pd
import torch
import warnings
from torch.utils.data import Dataset, DataLoader


class FeatsDataset(Dataset):
    def __init__(self, config, split="train"):
        self.config = config
        self.split = split

        self.root_path = self.config["root_path"]
        self.csv_root_path = self.config["csv_root_path"]

        self.df = pd.read_csv(os.path.join(self.csv_root_path, f"{self.split}_labels.csv"))
        self.feats_dir = os.path.join(self.root_path, self.split)
        if not os.path.exists(self.feats_dir):
            self.feats_dir = self.root_path

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        base = os.path.join(self.feats_dir, row["path"][:-4])
        try:
            feats = np.load(base + ".npz", allow_pickle=True)
        except FileNotFoundError:
            try:
                feats = np.load(base + ".npy", allow_pickle=True)
            except FileNotFoundError:
                warnings.warn(f"Features not found for {row['path']} in {self.feats_dir}")
                return None

        label = int(row["label"])

        if self.config["input_type"] == "audio":
            try:
                audio = feats['audio']
            except:
                try:
                    audio = feats['arr_0']
                except:
                    audio = feats
            video = -np.ones((audio.shape[0], 1024)) * np.inf
        elif self.config["input_type"] == "video":
            try:
                try:
                    video = feats['visual']
                except:
                    video = feats['video']
                if len(video.shape) > 2:
                    video = video.reshape(-1, video.shape[-1])
            except:
                try:
                    video = feats['arr_0']
                except:
                    video = feats
            audio = -np.ones((video.shape[0], 1024)) * np.inf
        elif self.config["input_type"] == "multimodal":
            try:
                video = feats["multimodal"]
                audio = feats["multimodal"]
            except:
                video = feats
                audio = feats
        else:
            raise ValueError(f"input_type should be multimodal, video or audio! Got: " + self.config["input_type"])

        if "apply_l2" in self.config and self.config["apply_l2"]:
            if self.config["input_type"] in ["both", "multimodal", "video"]:
                video = video / (np.linalg.norm(video, ord=2, axis=-1, keepdims=True))
            if self.config["input_type"] in ["both", "multimodal", "audio"]:
                audio = audio / (np.linalg.norm(audio, ord=2, axis=-1, keepdims=True))

        return torch.tensor(video, dtype=torch.float32), torch.tensor(audio, dtype=torch.float32), label, row["path"][:-4] + ".npz"  # video, audio, label, path


def load_data(config, test=False):

    if test:
        test_ds = FeatsDataset(config, split="test")
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=1)

        return test_dl

    else:
        train_ds = FeatsDataset(config, split="train")
        val_ds = FeatsDataset(config, split="val")

        def collate_skip_none(batch):
            batch = [b for b in batch if b is not None]
            if len(batch) == 0:
                # create a dummy batch instead of returning None
                return torch.utils.data.default_collate([(torch.empty(0), torch.empty(0), torch.empty(0), "")])
            return torch.utils.data.default_collate(batch)

        train_dl = DataLoader(train_ds, shuffle=True, batch_size=1, collate_fn=collate_skip_none, num_workers = 16)
        val_dl = DataLoader(val_ds, shuffle=False, batch_size=1, collate_fn=collate_skip_none, num_workers = 16)

        return train_dl, val_dl