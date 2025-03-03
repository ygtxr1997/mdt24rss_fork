import logging
import os
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader, Dataset
import torchvision

# from uha import make_pytorch_oxe_iterable_dataset, get_octo_dataset_tensorflow, get_single_dataset_tensorflow, multi_worker_iterable_dataset
from uha.uha_datamodule import UhaDataModule


logger = logging.getLogger(__name__)
DEFAULT_TRANSFORM = OmegaConf.create({"train": None, "val": None})
ONE_EP_DATASET_URL = "http://www.informatik.uni-freiburg.de/~meeso/50steps.tar.xz"


class OxeUhaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        # OriUhaDataModule
        transforms: DictConfig,  # Replace with your default transforms
        language_encoders: DictConfig,
        datasets: DictConfig,  # 'DATA_NAME','DATA_PATH','load_camera_views','action_xxx_norm','interleaved_cfg'
        batch_size: int = 4,
        num_workers: int = 0,  # will have no effect
        pin_memory: bool = True,
        drop_last: bool = True,
        # Others
        **kwargs: Dict,
    ):
        super().__init__()
        self.transforms_cfg = transforms
        self.language_encoders_cfg = language_encoders
        self.datasets_cfg = datasets

        self.ori_uha_cfg = OmegaConf.create({
            'datasets': datasets,
            'transforms': transforms,
            'language_encoders': language_encoders,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'drop_last': drop_last,
        })  # merge configs

        self.batch_size = batch_size
        self.modalities = ['lang']  # lang only

    def prepare_data(self, *args, **kwargs):
        pass
        print('[DEBUG] OxeUhaDataModule prepare_data finished.')

    def setup(self, stage=None):
        """
        Called by trainer.fit()
        """
        cfg = self.ori_uha_cfg
        is_main_process = os.environ.get("LOCAL_RANK", "0") == "0"

        ''' 1. Get dataloaders '''
        self.ori_uha_data_module = UhaDataModule(
            datasets=self.ori_uha_cfg['datasets'],
            batch_size=self.ori_uha_cfg['batch_size'],
            num_workers=self.ori_uha_cfg['num_workers'],
            pin_memory=self.ori_uha_cfg['pin_memory'],
            drop_last=self.ori_uha_cfg['drop_last'],
            transforms=self.ori_uha_cfg['transforms'],
            language_encoders=self.ori_uha_cfg['language_encoders'],
        )
        self.train_loader = self.ori_uha_data_module.create_train_dataloader(main_process=is_main_process)
        self.val_loader = self.ori_uha_data_module.create_val_dataloader()

        ''' 2. Get dataset info '''
        self.dataset_info = self.ori_uha_data_module.get_dataset_statistics()

        print(f'[DEBUG] OxeUhaDataModule setup finished (main={is_main_process}). '
              f'Train len={len(self.train_loader)}, '
              f'Val len={len(self.val_loader)}. Info: {self.dataset_info}')

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
