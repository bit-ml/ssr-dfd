import logging
import os

import hydra
from hydra.utils import instantiate
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
import torch

from data.data_module import DataModule
from finetune_learner import Learner


# static vars
os.environ["WANDB_SILENT"] = "true"
logging.getLogger("lightning").propagate = False


@hydra.main(config_path="conf", config_name="config_test")
def main(cfg):
    if cfg.fix_seed:
        seed_everything(42, workers=True)

    cfg.gpus = torch.cuda.device_count()
    print("num gpus:", cfg.gpus)

    data_module = DataModule(cfg)
    learner = Learner(cfg, output_dir=cfg.output_dir)

    trainer = Trainer(
        **cfg.trainer,
        strategy=DDPStrategy(find_unused_parameters=False) if cfg.gpus > 1 else None
    )

    trainer.test(learner, datamodule=data_module)


if __name__ == "__main__":
    main()
