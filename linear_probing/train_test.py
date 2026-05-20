import argparse
import random
import tqdm
import os
import yaml

from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from sklearn.metrics import average_precision_score, roc_auc_score

from datasets import load_data
from model import LinearModel


def init_callbacks(config):
    # LOGGER
    logger_path = config["logger"]["log_path"]
    if config["logger"]["name"] == "tensorboard":
        logger = TensorBoardLogger(logger_path)
    elif config["logger"]["name"] == "csv":
        logger = CSVLogger(logger_path)
    else:
        raise ValueError(config["logger"]["name"] + " not yet implemented!")

    # CALLBACKS
    callbacks = []
    if "ckpt_args" in config:
        callbacks.append(
            ModelCheckpoint(
                monitor=config["ckpt_args"]["metric"],
                dirpath=config["ckpt_args"]["ckpt_dir"],
                filename="model-{epoch:02d}",
                mode=config["ckpt_args"]["mode"],
            )
        )

    if "early_stopping" in config:
        callbacks.append(
            EarlyStopping(
                monitor=config["early_stopping"]["metric"],
                mode=config["early_stopping"]["mode"],
                patience=config["early_stopping"]["patience"],
            )
        )

    return logger, callbacks


def set_seed(seed):
    print(f"Using seed: {seed}", flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_checkpoint_path_from_folder(folder):
    files = os.listdir(folder)
    files = [f for f in files if f.endswith(".ckpt")]
    assert len(files) == 1, "More than one checkpoint in the folder"
    return os.path.join(folder, files[0])


def get_checkpoint_path(config):
    """Allows to get the checkpoint path from a training config."""
    if "ckpt_path" in config:
        return config["ckpt_path"]
    elif "callbacks" in config:
        checkpoint_dir = config["callbacks"]["ckpt_args"]["ckpt_dir"]
        return get_checkpoint_path_from_folder(checkpoint_dir)
    else:
        raise ValueError("No checkpoint path found in config.")


def get_output_path(config):
    """Allows to get the output path from a training config."""
    if "output_path" in config:
        return config["output_path"]
    elif "callbacks" in config:
        log_dir = Path(config["callbacks"]["logger"]["log_path"])
        out_dir = log_dir.parent
        out_dir = out_dir / "test"
        return str(out_dir)
    else:
        raise ValueError("No output path found in config.")


def train(config):
    train_dl, val_dl = load_data(config=config["data_info"])
    model = LinearModel(config=config)
    logger, callbacks = init_callbacks(config=config["callbacks"])

    trainer = L.Trainer(max_epochs=config["epochs"], logger=logger, callbacks=callbacks)
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


def test1(dataloader, path_checkpoint, path_output):
    model = LinearModel.load_from_checkpoint(path_checkpoint)
    model.to("cuda")
    model.eval()

    all_scores = np.array([])
    all_labels = np.array([])
    all_paths = np.array([])

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            video_feats, audio_feats, labels, paths = batch
            video_feats, audio_feats = video_feats.to("cuda"), audio_feats.to("cuda")

            scores = model.predict_scores(video_feats, audio_feats)

            all_scores = np.concatenate((all_scores, scores.cpu().numpy()), axis=0)
            all_labels = np.concatenate((all_labels, labels.cpu().numpy()), axis=0)
            all_paths = np.concatenate((all_paths, paths), axis=0)

    os.makedirs(path_output, exist_ok=True)

    pd.DataFrame(
        {"paths": all_paths, "scores": all_scores, "labels": all_labels}
    ).to_csv(os.path.join(path_output, "results.csv"), index=False)

    with open(os.path.join(path_output, "eval_results.txt"), "w") as f:
        f.write(f"AUC: {roc_auc_score(y_score=all_scores, y_true=all_labels)}\n")
        f.write(
            f"AP: {average_precision_score(y_score=all_scores, y_true=all_labels)}\n"
        )


def test(config):
    test_dl = load_data(config=config["data_info"], test=True)
    path_checkpoint = get_checkpoint_path(config)
    path_output = get_output_path(config)
    test1(
        dataloader=test_dl,
        path_checkpoint=path_checkpoint,
        path_output=path_output,
    )

    with open(os.path.join(path_output, "tested_config.yaml"), "w") as f:
        yaml.safe_dump(config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training and testing loop")

    parser.add_argument("--config_path")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    if args.test:
        test(config=config)
    else:
        train(config=config)
