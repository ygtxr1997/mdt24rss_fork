import logging
from pathlib import Path
import sys
from typing import List, Union
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


def print_leaf(prefix, x):
    if isinstance(x, str):
        print(f'{prefix}:{type(x)},len={len(x)}')
    elif isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        print(f'{prefix},{type(x)},shape={x.shape}')


def print_batch(prefix, x, depth=0):
    if isinstance(x, str):
        print_leaf(prefix, x)
        return
    elif isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        print_leaf(prefix, x)
        return
    elif isinstance(x, dict):
        print(f'{prefix}:Dict,keys={x.keys()}')
        for k, v in x.items():
            if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                print_leaf(('-' * depth) + k, v)
            else:
                print_batch(('-' * depth) + k, v, depth + 1)
    elif isinstance(x, list):
        print(f'List,len={len(x)},elem:{type(x[0])}')
        if isinstance(x[0], torch.Tensor) or isinstance(x[0], np.ndarray):
            print_batch(('-' * depth) + '[0]', x[0], depth + 1)
    else:
        raise TypeError(f'type {type(x)} not supported. x must be torch.Tensor or list or dict')


@hydra.main(config_path="../conf", config_name="config_abc_hk")
def main(cfg: DictConfig) -> None:
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    print('[DEBUG] datamodule loaded')
    datamodule.setup()

    for dataset_key, loader in datamodule.train_dataloader().items():
        print(('=' * 20) + f' Dataset {dataset_key} ' + ('=' * 20))
        for idx, example in enumerate(loader):
            if idx >= 20:
                break
            else:
                print_batch(f'Batch@{idx}th', example)
            print(('-' * 20) + ' Batch End ' + ('-' * 20))


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    os.environ["TOKENIZERS_PARALLELISM"] = 'True'
    os.environ["WANDB__SERVICE_WAIT"] = "300"  # alleviate ServiceStartTimeoutError
    main()
