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
from torch.utils.data import Dataset, DataLoader
from torch.nn import Conv2d
from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer, LightningDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
import litdata as ld


import mdt.models.mdt_agent as models_m
from mdt.utils.utils import (
    get_git_commit_hash,
    get_last_checkpoint,
    initialize_pretrained_weights,
    print_system_env_info,
)
from debug.de_dataloader import print_batch, print_leaf


class DebugDataset(Dataset):
    def __init__(self, modal: str = 'vis'):
        super().__init__()
        if modal == 'vis':
            self.data = torch.randn((1000, 3, 112, 112))
        else:
            self.data = torch.randn((500, 3, 112, 112))
        self.train_ep_npz_names = [int(x.split('_')[1].split('.')[0]) for x in os.listdir('/home/geyuan/datasets/CALVIN/dataset/task_D_D/training') if 'episode' in x]
        self.train_dict_ep_idx_to_dataset_order = np.load(os.path.join('/home/geyuan/datasets/CALVIN/dataset/litdata/task_D_D/training', "dict_ep_idx_to_dataset_order.npy"),
            allow_pickle=True).reshape(-1)[0]
        self.lit_train_dataset = DebugLitTrainDataset()

        self.val_ep_npz_names = [int(x.split('_')[1].split('.')[0]) for x in
                                   os.listdir('/home/geyuan/datasets/CALVIN/dataset/task_D_D/validation') if
                                   'episode' in x]
        self.val_dict_ep_idx_to_dataset_order = np.load(
            os.path.join('/home/geyuan/datasets/CALVIN/dataset/litdata/task_D_D/validation',
                         "dict_ep_idx_to_dataset_order.npy"),
            allow_pickle=True).reshape(-1)[0]
        self.lit_val_dataset = DebugLitValDataset()
        self.streaming_dataset = DebugStreamingDataset()
        self.pytorch_dataset = DebugPytorchDataset()
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        # print(f'[DEBUG][LitData] Rank@{os.environ["LOCAL_RANK"]}: len={len(self.lit_train_dataset)}')
        print(f'[DEBUG][StreamingDataset] Rank@{os.environ["LOCAL_RANK"]}: len={len(self.streaming_dataset)}')
        print(f'[DEBUG][PytorchDataset] Rank@{os.environ["LOCAL_RANK"]}: len={len(self.pytorch_dataset)}')

        # self.lit_train_dataset.__getitem__(self.dict_ep_idx_to_dataset_order[self.ep_npz_names[idx]])
        # print(f'[DEBUG] __getitem__train Rank@{os.environ["LOCAL_RANK"]}: {self.dict_ep_idx_to_dataset_order[self.ep_npz_names[idx]]} got!')

        # train_indices = np.random.randint(0, 2 * len(self.lit_train_dataset), size=1).tolist()
        # [self.lit_train_dataset.__getitem__(x) for x in train_indices]
        # print(f'[DEBUG] __getitem__train Rank@{os.environ["LOCAL_RANK"]}: {train_indices} got!')

        # self.lit_val_dataset.__getitem__(2 * len(self.lit_val_dataset))
        # print(f'[DEBUG] __getitem__val Rank@{os.environ["LOCAL_RANK"]}: {2 * len(self.lit_val_dataset)} got!')

        self.lit_train_dataset.__getitem__(self.val_dict_ep_idx_to_dataset_order[self.val_ep_npz_names[idx]])
        print(f'[DEBUG] __getitem__train Rank@{os.environ["LOCAL_RANK"]}: {self.val_dict_ep_idx_to_dataset_order[self.val_ep_npz_names[idx]]} got!')
        return {
            "data": self.data[idx],
            "idx": idx
        }


class DebugLitTrainDataset(ld.StreamingDataset):
    def __init__(self, dir='/home/geyuan/datasets/CALVIN/dataset/litdata/task_D_D/training'):
        super().__init__(
            input_dir=f'local:{dir}',
            max_cache_size="10GB",
            shuffle=False,
            drop_last=False
        )
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return {
            "data": data,
            "idx": idx
        }


class DebugLitValDataset(ld.StreamingDataset):
    def __init__(self, dir='/home/geyuan/datasets/CALVIN/dataset/litdata/task_D_D/validation'):
        super().__init__(
            input_dir=f'local:{dir}',
            max_cache_size="10GB",
            shuffle=False,
            drop_last=False
        )
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return {
            "data": data,
            "idx": idx
        }


class DebugStreamingDataset(ld.StreamingDataset):
    def __init__(self, dir='/home/geyuan/datasets/CALVIN/dataset/litdata/calvin_debug_dataset/training'):
        super().__init__(
            input_dir=f'local:{dir}',
            max_cache_size="10GB",
            shuffle=False,
            drop_last=False
        )
        self.data = torch.arange(0, 1000)
    def __getitem__(self, idx):
        data = self.data[idx]
        return {
            "data": data,
            "idx": idx
        }
    def __len__(self):
        return self.data.shape[0]


class DebugPytorchDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = torch.arange(0, 1000)
    def __getitem__(self, idx):
        data = self.data[idx]
        return {
            "data": data,
            "idx": idx
        }
    def __len__(self):
        return self.data.shape[0]


class DebugModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = Conv2d(3, 3, 3, 1, 1)
        print('[DEBUG] Model loaded.')
    def forward(self, x):
        return self.layer(x)
    def training_step(self, batch, batch_idx):
        if 'robot_obs' in batch["vis"].keys():
            data = batch["vis"]["robot_obs"]
        else:
            data = batch["vis"]["data"]
        data = data.mean() * (torch.randn(1, 3, 112, 112).to(data.device))

        data_idx = batch["vis"]["idx"]
        if int(data_idx) == 0:
            print(f'[DEBUG] Rank@{os.environ["LOCAL_RANK"]}: {batch_idx} indices={data_idx}')
        return self.forward(data).mean()
    def configure_optimizers(self):
        print('[DEBUG] Optimizer configured.')
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class DebugDatamodule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.dataset = {}
    def setup(self, stage=None):
        self.dataset['vis'] = DebugDataset('vis')
        self.dataset['lang'] = DebugDataset('lang')
        print(f'[DEBUG] Rank@{os.environ["LOCAL_RANK"]} setup: dataset_len={len(self.dataset["vis"])}')
    def train_dataloader(self):
        print(f'[DEBUG] Rank@{os.environ["LOCAL_RANK"]} train_dataloader: dataset_len={len(self.dataset["vis"])}')
        return {
            k: DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
            for k, dataset in self.dataset.items()
        }


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