import logging
from pathlib import Path
import sys
from typing import List, Union
import os
import wandb

# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())
import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
import torch
from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only


import mdt.models.mdt_agent as models_m
from mdt.utils.utils import (
    get_git_commit_hash,
    get_last_checkpoint,
    initialize_pretrained_weights,
    print_system_env_info,
)


@hydra.main(config_path="../conf", config_name="config_abc_hk")
def main(cfg: DictConfig) -> None:
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    print('datamodule loaded')


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    os.environ["TOKENIZERS_PARALLELISM"] = 'True'
    os.environ["WANDB__SERVICE_WAIT"] = "300"  # alleviate ServiceStartTimeoutError
    main()
