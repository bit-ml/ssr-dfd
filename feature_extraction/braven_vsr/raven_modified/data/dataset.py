import os

import cv2
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


def cut_or_pad(data, size, dim=0):
    # Pad with zeros on the right if data is too short
    if data.size(dim) < size:
        # assert False
        padding = size - data.size(dim)
        data = torch.from_numpy(np.pad(data, (0, padding), "constant"))
    # Cut from the right if data is too long
    elif data.size(dim) > size:
        data = data[:size]
    # Keep if data is exactly right
    assert data.size(dim) == size
    return data


class AVDataset(Dataset):
    def __init__(
        self,
        data_path,
        path_prefix,
        video_path_prefix_lrs2,
        audio_path_prefix_lrs2,
        video_path_prefix_lrs3,
        audio_path_prefix_lrs3,
        video_path_prefix_vox2=None,
        audio_path_prefix_vox2=None,
        transforms=None,
        modality="audiovisual",
    ):
        self.data_path = data_path
        self.path_prefix = path_prefix
        self.video_path_prefix_lrs3 = video_path_prefix_lrs3
        self.audio_path_prefix_lrs3 = audio_path_prefix_lrs3
        self.video_path_prefix_vox2 = video_path_prefix_vox2
        self.audio_path_prefix_vox2 = audio_path_prefix_vox2
        self.video_path_prefix_lrs2 = video_path_prefix_lrs2
        self.audio_path_prefix_lrs2 = audio_path_prefix_lrs2
        self.transforms = transforms
        self.modality = modality

        self.paths_counts_labels = self.configure_files()
        self.num_fails = 0

    def configure_files(self):
        paths_counts_labels = []
        with open(self.data_path, "r") as f:
            for path_count_label in f.read().splitlines():
                try:
                    file_path, label = path_count_label.split(",")
                except:
                    file_path, label, _ = path_count_label.split(",")
                if label == "label":
                    continue

                paths_counts_labels.append(
                    (file_path, [int(lab) for lab in label.split()])
                )
        return paths_counts_labels

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                break
        cap.release()
        if not frames:
            print(path)
            return None
        frames = torch.from_numpy(np.stack(frames))
        frames = frames.permute((3, 0, 1, 2))  # TxHxWxC -> # CxTxHxW
        return frames

    def load_audio(self, path):
        audio, org_sr = torchaudio.load(path, normalize=True)
        if audio.shape[0] > 1:
            print(f"Dual channel for: {path}; Changing to mono..", flush=True)
            audio = audio.mean(dim=0, keepdim=True)
        if org_sr != 16000:
            print(f"Different sampling size for: {path}; Resampling..", flush=True)
            audio = torchaudio.functional.resample(audio, orig_freq=org_sr, new_freq=16000)
        return audio

    def __len__(self):
        return len(self.paths_counts_labels)

    def __getitem__(self, index):
        file_path, label = self.paths_counts_labels[index]

        if self.modality == "video":
            data = self.load_video(os.path.join(self.path_prefix, file_path[:-4] + "_roi.mp4"))
            if data is None:
                self.num_fails += 1
                if self.num_fails == 200:
                    raise ValueError("Too many file errors.")
                return {"data": None, "label": None, "filepath": file_path}
            data = self.transforms["video"](data).permute((1, 2, 3, 0))
        elif self.modality == "audio":
            data = self.load_audio(
                os.path.join(self.path_prefix, file_path[:-4] + ".wav")
            )
            data = self.transforms["audio"](data).squeeze(0)
        elif self.modality == "audiovisual":
            video = self.load_video(os.path.join(self.path_prefix, file_path[:-4] + "_roi.mp4"))
            if video is None:
                self.num_fails += 1
                if self.num_fails == 200:
                    raise ValueError("Too many file errors.")
                return {"video": None, "audio": None, "label": None}
            audio = self.load_audio(
                os.path.join(self.path_prefix, file_path[:-4] + ".wav")
            )
            audio = cut_or_pad(audio.squeeze(0), video.size(1) * 640)
            video = self.transforms["video"](video).permute((1, 2, 3, 0))
            audio = self.transforms["audio"](audio.unsqueeze(0)).squeeze(0)
            return {"video": video, "audio": audio, "label": torch.tensor(label)}

        return {"data": data, "label": torch.tensor(label), "filepath": file_path}
