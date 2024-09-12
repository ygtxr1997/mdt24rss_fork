import logging
from pathlib import Path
import sys
from typing import List, Union, Dict
import os

import numpy as np
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
from debug.de_dataloader import print_batch, print_leaf


@hydra.main(config_path="../conf", config_name="config_d_hk")
def main(cfg: DictConfig) -> None:
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    print('[DEBUG] datamodule loaded')
    datamodule.setup()
    train_set: Dict = datamodule.train_datasets

    print(f'[DEBUG] lang_len={len(train_set["lang"])}, vis_len={len(train_set["vis"])}')
    print_batch('--- Lang batch', train_set["lang"].__getitem__(0))
    print_batch('--- Vis batch', train_set["vis"][0])

    # print_batch('Dataset.episode_lookup', train_set["vis"].episode_lookup)
    # print_batch('Dataset.lang_lookup', train_set["vis"].lang_lookup)
    # print_batch('Dataset.lang_ann', train_set["vis"].lang_ann)
    # print_batch('Dataset.lang_text', train_set["vis"].lang_text)


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    os.environ["TOKENIZERS_PARALLELISM"] = 'True'
    os.environ["WANDB__SERVICE_WAIT"] = "300"  # alleviate ServiceStartTimeoutError
    main()