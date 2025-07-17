import logging
import os
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
from PIL import Image
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader, Dataset
import torchvision

# from uha import make_pytorch_oxe_iterable_dataset, get_octo_dataset_tensorflow, get_single_dataset_tensorflow, multi_worker_iterable_dataset
from uha.uha_datamodule import UhaDataModule

from mdt.utils.transforms import RandomShiftsAug, NormalizeVector


class ImageDataset(Dataset):
    """
    A custom dataset class to read images from a directory with two-level subdirectories.
    """

    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (string): Path to the root directory containing images and subdirectories.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = image_dir
        self.transform = transform
        # Gather the paths to all images in the two-level subdirectory structure
        self.image_paths = []

        # Traverse the directory to find all image files within two-level subdirectories
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))

        print(f"[ImageDataset] root={self.image_dir}, images={len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load an image from the dataset
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Convert image to RGB format
        zero_pad = Image.fromarray(np.zeros((84, 84, 3), dtype=np.uint8))

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
            zero_pad = torchvision.transforms.ToTensor()(zero_pad)  # to [0,1]
            zero_pad = NormalizeVector(mean=0.5, std=0.5)(zero_pad)  # to [-1,1]

            # add frame dim
            image = image.unsqueeze(0)
            zero_pad = zero_pad.unsqueeze(0)

        return {
            'observation': {
                'image_primary': image,  # in [-1,1]
                'image_wrist': zero_pad,  # in [-1,1], all -1
            }
        }


class OxeUhaDomainAdaptDataModule(pl.LightningDataModule):
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
            # Domain Adaptation
            target_data_dir: str = "",
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

        # For target
        self.t_data_dir = target_data_dir
        assert os.path.exists(self.t_data_dir), f"{self.t_data_dir} does not exist"
        self.t_train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.RandomCrop(224, padding=10),
            torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03),
            torchvision.transforms.ToTensor(),  # to [0,1]
            NormalizeVector(mean=0.5, std=0.5),  # to [-1,1]
        ])
        self.t_val_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),  # to [0,1]
            NormalizeVector(mean=0.5, std=0.5),  # to [-1,1]
        ])

    def prepare_data(self, *args, **kwargs):
        pass
        print('[DEBUG] OxeUhaDomainAdaptDataModule prepare_data finished.')

    def setup(self, stage=None):
        """
        Called by trainer.fit()
        """
        cfg = self.ori_uha_cfg
        is_main_process = os.environ.get("LOCAL_RANK", "0") == "0"

        ''' 1. Get dataloaders '''
        # Source
        self.ori_uha_data_module = UhaDataModule(
            datasets=self.ori_uha_cfg['datasets'],
            batch_size=self.ori_uha_cfg['batch_size'],
            num_workers=self.ori_uha_cfg['num_workers'],
            pin_memory=self.ori_uha_cfg['pin_memory'],
            drop_last=self.ori_uha_cfg['drop_last'],
            transforms=self.ori_uha_cfg['transforms'],
            language_encoders=self.ori_uha_cfg['language_encoders'],
        )
        self.s_train_loader = self.ori_uha_data_module.create_train_dataloader(main_process=is_main_process)
        self.s_val_loader = self.ori_uha_data_module.create_val_dataloader()

        # Target
        t_train_dataset = ImageDataset(self.t_data_dir, transform=self.t_train_transform)
        t_val_dataset = ImageDataset(self.t_data_dir, transform=self.t_val_transform)
        self.t_train_loader = DataLoader(
            t_train_dataset, batch_size=self.batch_size,
            num_workers=4, pin_memory=True,
            shuffle=True, drop_last=True,
        )
        self.t_val_loader = DataLoader(
            t_val_dataset, batch_size=self.batch_size,
            num_workers=4, pin_memory=True,
        )

        ''' 2. Get dataset info '''
        self.dataset_info = self.ori_uha_data_module.get_dataset_statistics()

        print(f'[DEBUG] OxeUhaDomainAdaptDataModule setup finished (main={is_main_process}). '
              f'Source Train len={len(self.s_train_loader)}, '
              f'Source Val len={len(self.s_val_loader)}.'
              f'Target Train len={len(self.t_train_loader)}, Target Val len={len(self.t_val_loader)}.')

    def train_dataloader(self):
        """
        OXE_BATCH: Dict,keys=dict_keys(['source', 'target'])
        source: Dict,keys=dict_keys(['observation', 'task', 'action', 'action_pad_mask'])
        -observation: Dict,keys=dict_keys(['image_primary', 'image_secondary', 'image_wrist', 'timestep', 'pad_mask_dict', 'timestep_pad_mask', 'task_completed'])
        --image_primary,<class 'torch.Tensor'>,shape=torch.Size([128, 1, 3, 224, 224]),min=0.00,max=255.00
        --image_secondary,<class 'torch.Tensor'>,shape=torch.Size([128, 1, 3, 224, 224]),min=0.00,max=0.00
        --image_wrist,<class 'torch.Tensor'>,shape=torch.Size([128, 1, 3, 84, 84]),min=0.00,max=0.00
        --timestep,<class 'torch.Tensor'>,shape=torch.Size([128, 1]),min=0.00,max=162.00
        --pad_mask_dict: Dict,keys=dict_keys(['image_primary', 'image_secondary', 'image_wrist', 'timestep'])
        ---image_primary,<class 'torch.Tensor'>,shape=torch.Size([128, 1]),min=1.00,max=1.00
        ---image_secondary,<class 'torch.Tensor'>,shape=torch.Size([128, 1]),min=0.00,max=0.00
        ---image_wrist,<class 'torch.Tensor'>,shape=torch.Size([128, 1]),min=0.00,max=0.00
        ---timestep,<class 'torch.Tensor'>,shape=torch.Size([128, 1]),min=1.00,max=1.00
        --timestep_pad_mask,<class 'torch.Tensor'>,shape=torch.Size([128, 1]),min=1.00,max=1.00
        --task_completed,<class 'torch.Tensor'>,shape=torch.Size([128, 1, 10]),min=0.00,max=1.00
        -task: Dict,keys=dict_keys(['language_instruction', 'language_key', 'pad_mask_dict'])
        --language_instruction: List,len=1,elem:<class 'list'>
        --language_key,<class 'torch.Tensor'>,shape=torch.Size([128]),min=0.00,max=0.00
        --pad_mask_dict: Dict,keys=dict_keys(['language_instruction', 'language_key'])
        ---language_instruction,<class 'torch.Tensor'>,shape=torch.Size([128]),min=1.00,max=1.00
        ---language_key,<class 'torch.Tensor'>,shape=torch.Size([128]),min=1.00,max=1.00
        -action,<class 'torch.Tensor'>,shape=torch.Size([128, 1, 10, 7]),min=-1.00,max=1.00
        -action_pad_mask,<class 'torch.Tensor'>,shape=torch.Size([128, 1, 10, 7]),min=0.00,max=1.00
        target: Dict,keys=dict_keys(['observation'])
        -observation: Dict,keys=dict_keys(['image_primary', 'image_wrist'])
        --image_primary,<class 'torch.Tensor'>,shape=torch.Size([128, 1, 3, 224, 224]),min=-1.00,max=1.00
        --image_wrist,<class 'torch.Tensor'>,shape=torch.Size([128, 1, 3, 84, 84]),min=-1.00,max=-1.00
        """
        return CombinedLoader({
            'source': self.s_train_loader,
            'target': self.t_train_loader,
        }, "min_size")

    def val_dataloader(self):
        return CombinedLoader({
            'source': self.s_val_loader,
            'target': self.t_val_loader,
        }, "min_size")
