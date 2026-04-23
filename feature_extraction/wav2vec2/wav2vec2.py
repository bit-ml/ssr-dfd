import argparse
import os

import numpy as np
import pandas as pd
import tqdm
import torch
from transformers import AutoFeatureExtractor, WavLMModel, Wav2Vec2Model
import torchaudio


class HuggingFaceFeatureExtractor:
    def __init__(self, model_class, name):
        self.device = "cuda"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(name)
        self.model = model_class.from_pretrained(name)
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, audio, sr):
        inputs = self.feature_extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            # padding=True,
            # max_length=16_000,
            # truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                # output_attentions=True,
                # output_hidden_states=False,
            )
        return outputs.last_hidden_state


FEATURE_EXTRACTORS = {
    "wav2vec2-base": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-base"
    ),
    "wav2vec2-large": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-large"
    ),
    "wav2vec2-large-lv60": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-large-lv60"
    ),
    "wav2vec2-large-robust": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-large-robust"
    ),
    "wav2vec2-large-xlsr-53": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-large-xlsr-53"
    ),
    "wav2vec2-xls-r-300m": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-xls-r-300m"
    ),
    "wav2vec2-xls-r-1b": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-xls-r-1b"
    ),
    "wav2vec2-xls-r-2b": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-xls-r-2b"
    ),
    "wavlm-base": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "microsoft/wavlm-base"
    ),
    "wavlm-base-sv": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "microsoft/wavlm-base-sv"
    ),
    "wavlm-base-plus": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "microsoft/wavlm-base-plus"
    ),
    "wavlm-large": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "microsoft/wavlm-large"
    ),
}
SAMPLING_RATE = 16_000
WAV2VEC_MODEL_NAME = "wav2vec2-xls-r-2b"


def load_wav2vec(path, feature_extractor):
    def extract1(audio, feature_extractor):
        feature = feature_extractor(audio, sr=SAMPLING_RATE)
        feature = feature.cpu().numpy().squeeze()
        return feature

    try:
        audio, sr = torchaudio.load(path)
    except:
        print(f"ERROR: {path}")
        return None

    # Convert to mono (if stereo)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    # Resample if needed
    if sr != SAMPLING_RATE:
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLING_RATE)(audio)
    audio = audio.squeeze(0)

    try:
        feature = extract1(audio, feature_extractor)
    except Exception as e:
        print(e)
        print(f"ERROR OOM: {path}")
        return None

    if len(feature) % 2 != 0:
        feature = np.vstack([feature, feature[-1]])

    feature_new = feature.reshape(len(feature) // 2, 2, feature.shape[1]).reshape(len(feature) // 2, feature.shape[1] * 2)
    print(f"Audio: {feature_new.shape[0]}; Original sampling rate: {sr}", flush=True)

    return feature_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Audio feats extraction (using Wav2Vec2)'
    )

    parser.add_argument('--in_root_path', required=True)
    parser.add_argument('--out_root_path', required=True)
    parser.add_argument('--csv_file', default=None)
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    if args.csv_file is None:
        files = os.listdir(args.in_root_path)
        df = pd.DataFrame({
            "path": np.array(files)
        })
    else:
        df = pd.read_csv(args.csv_file)
        if 'path' not in df.columns:
            raise ValueError("path column is required in the csv file")

    feature_type = WAV2VEC_MODEL_NAME
    feature_extractor = FEATURE_EXTRACTORS[feature_type]()
    feature_type = feature_type.replace("wav2vec2-", "")

    if args.test:
        paths = []
        audios = []

    for idx, row in tqdm.tqdm(df.iterrows()):
        src = os.path.join(args.in_root_path, row['path'][:-4] + ".wav")
        dst = os.path.join(args.out_root_path, row['path'][:-4] + ".npy")

        audio_feats = load_wav2vec(src, feature_extractor)
        if audio_feats is None:
            continue
        if args.test:
            paths.append(row['path'])
            audios.append(audio_feats)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        np.save(dst, audio_feats)

    if args.test:
        os.makedirs(args.out_root_path, exist_ok=True)
        np.save(os.path.join(args.out_root_path, "paths.npy"), np.array(paths, dtype=object))
        np.save(os.path.join(args.out_root_path, "audio.npy"), np.array(audios, dtype=object)) 
