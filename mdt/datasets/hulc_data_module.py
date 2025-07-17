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

import mdt
from mdt.datasets.utils.episode_utils import load_dataset_statistics
from mdt.datasets.utils.shared_memory_utils import load_shm_lookup, save_shm_lookup, SharedMemoryLoader
from mdt.datasets.tcl_dataset import TCLImageDataset

logger = logging.getLogger(__name__)
DEFAULT_TRANSFORM = OmegaConf.create({"train": None, "val": None})
ONE_EP_DATASET_URL = "http://www.informatik.uni-freiburg.de/~meeso/50steps.tar.xz"


class ConcatenatedDataset(Dataset):
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.dataset_lens = [d.chose_len for d in datasets]
        self.batch_size = datasets[0].batch_size
        self.num_workers = datasets[0].num_workers
        self.total_len = sum(self.dataset_lens)
        self.dataset_len_prefix_sum = []
        for d_len in self.dataset_lens:
            if len(self.dataset_len_prefix_sum) == 0:
                self.dataset_len_prefix_sum.append(d_len)
            else:
                self.dataset_len_prefix_sum.append(self.dataset_len_prefix_sum[-1] + d_len)
        print(f"[DEBUG] ConcatenatedDataset, total_len={self.total_len}, split_lens={self.dataset_lens}")
    def __len__(self):
        return self.total_len
    def __getitem__(self, idx, **kwargs):
        previous_dataset_len_sum = 0
        for dataset_idx, dataset_len_sum in enumerate(self.dataset_len_prefix_sum):
            if idx < dataset_len_sum:
                relative_idx = idx - previous_dataset_len_sum
                # print(f"[DEBUG] {idx} : {relative_idx} in [{previous_dataset_len_sum}, {dataset_len_sum}]")
                return self.datasets[dataset_idx].__getitem__(relative_idx, **kwargs)
            else:
                previous_dataset_len_sum += dataset_len_sum
        raise IndexError(f"{idx} >= {self.total_len}")


class HulcDataModule(pl.LightningDataModule):
    """ A warping of uha """
    def __init__(
        self,
        datasets: DictConfig,
        root_data_dir: str = "data",
        num_workers: int = 8,
        transforms: DictConfig = DEFAULT_TRANSFORM,
        shuffle_val: bool = False,
        train_ratio: float = 1.,
        val_as_train_ratio: float = 0.,
        **kwargs: Dict,
    ):
        super().__init__()
        self.datasets_cfg = datasets
        self.train_datasets = None
        self.val_datasets = None
        self.train_sampler = None
        self.val_sampler = None
        self.num_workers = num_workers
        root_data_path = Path(root_data_dir)
        if not root_data_path.is_absolute():
            root_data_path = Path(mdt.__file__).parent / root_data_path
        self.training_dir = root_data_path / "training"
        self.val_dir = root_data_path / "validation"
        self.shuffle_val = shuffle_val
        self.modalities: List[str] = []
        self.transforms = transforms
        self.train_ratio = train_ratio
        self.val_as_train_ratio = val_as_train_ratio

        if 'lang_dataset' in self.datasets_cfg: 
            if "shm_dataset" in self.datasets_cfg.lang_dataset._target_:
                self.use_shm = "shm_dataset" in self.datasets_cfg.lang_dataset._target_
            else:
                self.use_shm = False
        elif 'shm_dataset' in self.datasets_cfg.vision_dataset._target_:
            self.use_shm = True
        else:
            self.use_shm = False

    def prepare_data(self, *args, **kwargs):
        # check if files already exist
        dataset_exist = np.any([len(list(self.training_dir.glob(extension))) for extension in ["*.npz", "*.pkl"]])

        # download and unpack images
        if not dataset_exist:
            if "CI" not in os.environ:
                print(f"No dataset found in {self.training_dir}.")
                print("For information how to download to full CALVIN dataset, please visit")
                print("https://github.com/mees/calvin/tree/main/dataset")
                print("Do you wish to download small debug dataset to continue training?")
                s = input("YES / no")
                if s == "no":
                    exit()
            logger.info(f"downloading dataset to {self.training_dir} and {self.val_dir}")
            torchvision.datasets.utils.download_and_extract_archive(ONE_EP_DATASET_URL, self.training_dir)
            torchvision.datasets.utils.download_and_extract_archive(ONE_EP_DATASET_URL, self.val_dir)

        if self.use_shm:
            # When using shared memory dataset, initialize lookups
            train_shmem_loader = SharedMemoryLoader(self.datasets_cfg, self.training_dir)
            train_shm_lookup = train_shmem_loader.load_data_in_shared_memory()

            val_shmem_loader = SharedMemoryLoader(self.datasets_cfg, self.val_dir)
            val_shm_lookup = val_shmem_loader.load_data_in_shared_memory()

            save_shm_lookup(train_shm_lookup, val_shm_lookup)
        print('[DEBUG] HulcDataModule prepare_data finished.')

    def setup(self, stage=None):
        """
        Called by trainer.fit()
        """
        transforms = load_dataset_statistics(self.training_dir, self.val_dir, self.transforms)

        # self.train_transforms = {
        #    cam: [hydra.utils.instantiate(transform) for transform in transforms.train[cam]] for cam in transforms.train
        #}
        self.train_transforms = {}
        for cam in transforms.train:
            # print("Processing camera:", cam)
            cam_transforms = []
            for transform in transforms.train[cam]:
                # print("Instantiating transform for camera", cam, ":", transform)
                if transform._target_ == "torchvision.transforms.ColorJitter":
                    instantiated_transform = torchvision.transforms.ColorJitter(
                        brightness=transform.brightness,
                        contrast=tuple(transform.contrast),
                        saturation=tuple(transform.saturation),
                    )
                else:
                    instantiated_transform = hydra.utils.instantiate(transform)
                cam_transforms.append(instantiated_transform)
            self.train_transforms[cam] = cam_transforms

        self.val_transforms = {
            cam: [hydra.utils.instantiate(transform) for transform in transforms.val[cam]] for cam in transforms.val
        }
        self.train_transforms = {key: torchvision.transforms.Compose(val) for key, val in self.train_transforms.items()}
        self.val_transforms = {key: torchvision.transforms.Compose(val) for key, val in self.val_transforms.items()}
        self.train_datasets, self.train_sampler, self.val_datasets, self.val_sampler = {}, {}, {}, {}

        if self.use_shm:
            train_shm_lookup, val_shm_lookup = load_shm_lookup()

        for _, dataset in self.datasets_cfg.items():
            if dataset == 'lang_paraphrase-MiniLM-L3-v2':
                continue
            else:
                train_dataset = hydra.utils.instantiate(
                    dataset, datasets_dir=self.training_dir, transforms=self.train_transforms,
                    chose_ratio=self.train_ratio,
                )
                if self.val_as_train_ratio > 0.:
                    val_as_train_dataset = hydra.utils.instantiate(
                        dataset, datasets_dir=self.val_dir, transforms=self.train_transforms,
                        chose_ratio=self.val_as_train_ratio,
                    )
                    train_dataset = ConcatenatedDataset([train_dataset, val_as_train_dataset])
                val_dataset = hydra.utils.instantiate(dataset, datasets_dir=self.val_dir, transforms=self.val_transforms)
                if self.use_shm:
                    train_dataset.setup_shm_lookup(train_shm_lookup)
                    val_dataset.setup_shm_lookup(val_shm_lookup)
                key = dataset.key
                self.train_datasets[key] = train_dataset
                self.val_datasets[key] = val_dataset
                self.modalities.append(key)
        print(f'[DEBUG] HulcDataModule setup finished. train_ratio={self.train_ratio * 100:.2f}%, '
              f'val_as_train_ratio={self.val_as_train_ratio * 100:.2f}%')

    def train_dataloader(self):
        return {
            key: DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                num_workers=dataset.num_workers,
                pin_memory=True,
                shuffle=True,
                prefetch_factor=2,
            )
            for key, dataset in self.train_datasets.items()
        }

    def test_dataloader(self):  # just for debug
        return {
            key: DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                num_workers=dataset.num_workers,
                pin_memory=True,
                shuffle=False,
            )
            for key, dataset in self.train_datasets.items()
        }

    def val_dataloader(self):
        val_dataloaders = {
            key: DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                num_workers=dataset.num_workers,
                pin_memory=True,
            )
            for key, dataset in self.val_datasets.items()
        }
        # combined_val_loaders = val_dataloaders['vis']
        combined_val_loaders = CombinedLoader(val_dataloaders, "max_size_cycle")
        return combined_val_loaders


class HulcTCLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        root_data_dir: str = "data",
        num_workers: int = 8,
        transforms: DictConfig = DEFAULT_TRANSFORM,
        shuffle_val: bool = False,
        train_ratio: float = 1.,
        val_as_train_ratio: float = 0.,
        **kwargs: Dict,
    ):
        super().__init__()
        self.datasets_cfg = datasets
        self.train_datasets = None
        self.val_datasets = None
        self.train_sampler = None
        self.val_sampler = None
        self.num_workers = num_workers
        root_data_path = Path(root_data_dir)
        if not root_data_path.is_absolute():
            root_data_path = Path(mdt.__file__).parent / root_data_path

        self.tcl_root = os.path.dirname(root_data_path)
        self.training_dir = root_data_path
        self.training_basename = os.path.basename(root_data_path)
        self.training_h5_path = os.path.join(self.tcl_root, "hdf5", self.training_basename + "_240p.h5")

        # self.val_dir = root_data_path / "validation"
        # self.shuffle_val = shuffle_val

        self.modalities: List[str] = []
        self.transforms = transforms
        self.train_ratio = train_ratio
        self.val_as_train_ratio = val_as_train_ratio

        if 'lang_dataset' in self.datasets_cfg:
            if "shm_dataset" in self.datasets_cfg.lang_dataset._target_:
                self.use_shm = "shm_dataset" in self.datasets_cfg.lang_dataset._target_
            else:
                self.use_shm = False
        elif 'shm_dataset' in self.datasets_cfg.vision_dataset._target_:
            self.use_shm = True
        else:
            self.use_shm = False

    def prepare_data(self, *args, **kwargs):
        # # check if files already exist
        # dataset_exist = np.any([len(list(self.training_dir.glob(extension))) for extension in ["*.npz", "*.pkl"]])
        #
        # # download and unpack images
        # if not dataset_exist:
        #     if "CI" not in os.environ:
        #         print(f"No dataset found in {self.training_dir}.")
        #         print("For information how to download to full CALVIN dataset, please visit")
        #         print("https://github.com/mees/calvin/tree/main/dataset")
        #         print("Do you wish to download small debug dataset to continue training?")
        #         s = input("YES / no")
        #         if s == "no":
        #             exit()
        #     logger.info(f"downloading dataset to {self.training_dir} and {self.val_dir}")
        #     torchvision.datasets.utils.download_and_extract_archive(ONE_EP_DATASET_URL, self.training_dir)
        #     torchvision.datasets.utils.download_and_extract_archive(ONE_EP_DATASET_URL, self.val_dir)
        #
        # if self.use_shm:
        #     # When using shared memory dataset, initialize lookups
        #     train_shmem_loader = SharedMemoryLoader(self.datasets_cfg, self.training_dir)
        #     train_shm_lookup = train_shmem_loader.load_data_in_shared_memory()
        #
        #     val_shmem_loader = SharedMemoryLoader(self.datasets_cfg, self.val_dir)
        #     val_shm_lookup = val_shmem_loader.load_data_in_shared_memory()
        #
        #     save_shm_lookup(train_shm_lookup, val_shm_lookup)
        # print('[DEBUG] HulcTCLDataModule prepare_data finished.')
        pass

    def setup(self, stage=None):
        """
        Called by trainer.fit()
        """
        # transforms = load_dataset_statistics(self.training_dir, self.val_dir, self.transforms)

        # self.train_transforms = {
        #    cam: [hydra.utils.instantiate(transform) for transform in transforms.train[cam]] for cam in transforms.train
        #}
        # self.train_transforms = {}
        # for cam in transforms.train:
        #     # print("Processing camera:", cam)
        #     cam_transforms = []
        #     for transform in transforms.train[cam]:
        #         # print("Instantiating transform for camera", cam, ":", transform)
        #         if transform._target_ == "torchvision.transforms.ColorJitter":
        #             instantiated_transform = torchvision.transforms.ColorJitter(
        #                 brightness=transform.brightness,
        #                 contrast=tuple(transform.contrast),
        #                 saturation=tuple(transform.saturation),
        #             )
        #         else:
        #             instantiated_transform = hydra.utils.instantiate(transform)
        #         cam_transforms.append(instantiated_transform)
        #     self.train_transforms[cam] = cam_transforms
        #
        # self.val_transforms = {
        #     cam: [hydra.utils.instantiate(transform) for transform in transforms.val[cam]] for cam in transforms.val
        # }
        # self.train_transforms = {key: torchvision.transforms.Compose(val) for key, val in self.train_transforms.items()}
        # self.val_transforms = {key: torchvision.transforms.Compose(val) for key, val in self.val_transforms.items()}
        self.train_datasets, self.train_sampler, self.val_datasets, self.val_sampler = {}, {}, {}, {}

        if self.use_shm:
            train_shm_lookup, val_shm_lookup = load_shm_lookup()

        for _, dataset in self.datasets_cfg.items():
            if dataset == 'lang_paraphrase-MiniLM-L3-v2':
                continue
            else:
                train_dataset = hydra.utils.instantiate(
                    dataset,
                    data_root=self.training_dir,
                    h5_path=self.training_h5_path,
                    chose_ratio=1.,
                    # chose_ratio=self.train_ratio,
                )
                # if self.val_as_train_ratio > 0.:
                #     val_as_train_dataset = hydra.utils.instantiate(
                #         dataset, datasets_dir=self.val_dir, transforms=self.train_transforms,
                #         chose_ratio=self.val_as_train_ratio,
                #     )
                #     train_dataset = ConcatenatedDataset([train_dataset, val_as_train_dataset])
                val_dataset = train_dataset.get_validation_dataset()

                if self.use_shm:
                    train_dataset.setup_shm_lookup(train_shm_lookup)
                    val_dataset.setup_shm_lookup(val_shm_lookup)

                key = dataset.key  # 'lang'
                self.train_datasets[key] = train_dataset
                self.val_datasets[key] = val_dataset
                self.modalities.append(key)
        print(f'[DEBUG] HulcTCLDataModule setup finished. train_ratio={self.train_ratio * 100:.2f}%, '
              f'val_as_train_ratio={self.val_as_train_ratio * 100:.2f}%')

    def train_dataloader(self):
        return {
            key: DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                num_workers=dataset.num_workers,
                pin_memory=True,
                shuffle=True,
                prefetch_factor=2,
            )
            for key, dataset in self.train_datasets.items()
        }

    def test_dataloader(self):  # just for debug
        return {
            key: DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                num_workers=dataset.num_workers,
                pin_memory=True,
                shuffle=False,
            )
            for key, dataset in self.train_datasets.items()
        }

    def val_dataloader(self):
        val_dataloaders = {
            key: DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                num_workers=dataset.num_workers,
                pin_memory=True,
            )
            for key, dataset in self.val_datasets.items()
        }
        # combined_val_loaders = val_dataloaders['vis']
        combined_val_loaders = CombinedLoader(val_dataloaders, "max_size_cycle")
        return combined_val_loaders


class HulcTCLMergeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        root_data_dirs: list = ("data",),  # Different (1)
        num_workers: int = 8,
        transforms: DictConfig = DEFAULT_TRANSFORM,
        shuffle_val: bool = False,
        train_ratio: float = 1.,
        val_as_train_ratio: float = 0.,
        **kwargs: Dict,
    ):
        super().__init__()

        self.datasets_cfg = datasets
        self.train_datasets = None
        self.val_datasets = None
        self.train_sampler = None
        self.val_sampler = None
        self.num_workers = num_workers

        self.training_dirs = []
        self.training_h5_paths = []

        for root_idx, root_data_dir in enumerate(root_data_dirs):
            root_data_path = Path(root_data_dir)
            if not root_data_path.is_absolute():
                root_data_path = Path(mdt.__file__).parent / root_data_path

            tcl_root = os.path.dirname(root_data_path)

            self.training_dirs.append(root_data_path)
            training_basename = os.path.basename(root_data_path)
            self.training_h5_paths.append(os.path.join(tcl_root, "hdf5", training_basename + "_240p.h5"))

        # self.val_dir = root_data_path / "validation"
        # self.shuffle_val = shuffle_val

        self.modalities: List[str] = []
        self.transforms = transforms
        self.train_ratio = train_ratio
        self.val_as_train_ratio = val_as_train_ratio

        if 'lang_dataset' in self.datasets_cfg:
            if "shm_dataset" in self.datasets_cfg.lang_dataset._target_:
                self.use_shm = "shm_dataset" in self.datasets_cfg.lang_dataset._target_
            else:
                self.use_shm = False
        elif 'shm_dataset' in self.datasets_cfg.vision_dataset._target_:
            self.use_shm = True
        else:
            self.use_shm = False

    def setup(self, stage=None):
        """
        Called by trainer.fit()
        """
        self.train_datasets, self.train_sampler, self.val_datasets, self.val_sampler = {}, {}, {}, {}

        if self.use_shm:
            train_shm_lookup, val_shm_lookup = load_shm_lookup()

        for _, dataset in self.datasets_cfg.items():
            if dataset == 'lang_paraphrase-MiniLM-L3-v2':
                continue
            else:
                train_dataset = hydra.utils.instantiate(
                    dataset,
                    data_roots=self.training_dirs,
                    h5_paths=self.training_h5_paths,
                    chose_ratio=1.,
                    # chose_ratio=self.train_ratio,
                )
                # if self.val_as_train_ratio > 0.:
                #     val_as_train_dataset = hydra.utils.instantiate(
                #         dataset, datasets_dir=self.val_dir, transforms=self.train_transforms,
                #         chose_ratio=self.val_as_train_ratio,
                #     )
                #     train_dataset = ConcatenatedDataset([train_dataset, val_as_train_dataset])
                val_dataset = train_dataset.get_validation_dataset()

                # Save meta statistics data
                meta_json_path = os.path.join(Path.cwd(), "checkpoints", "statistics.json")
                train_dataset.save_meta(meta_json_path)

                # if self.use_shm:
                #     train_dataset.setup_shm_lookup(train_shm_lookup)
                #     val_dataset.setup_shm_lookup(val_shm_lookup)

                key = dataset.key  # 'lang'
                self.train_datasets[key] = train_dataset
                self.val_datasets[key] = val_dataset
                self.modalities.append(key)
        print(f'[DEBUG] HulcTCLMergeDataModule setup finished. train_ratio={self.train_ratio * 100:.2f}%, '
              f'val_as_train_ratio={self.val_as_train_ratio * 100:.2f}%. training_dirs={self.training_dirs}')

    def train_dataloader(self):
        return {
            key: DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                num_workers=dataset.num_workers,
                pin_memory=True,
                shuffle=True,
                prefetch_factor=2,
            )
            for key, dataset in self.train_datasets.items()
        }

    def test_dataloader(self):  # just for debug
        return {
            key: DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                num_workers=dataset.num_workers,
                pin_memory=True,
                shuffle=False,
            )
            for key, dataset in self.train_datasets.items()
        }

    def val_dataloader(self):
        val_dataloaders = {
            key: DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                num_workers=dataset.num_workers,
                pin_memory=True,
            )
            for key, dataset in self.val_datasets.items()
        }
        # combined_val_loaders = val_dataloaders['vis']
        combined_val_loaders = CombinedLoader(val_dataloaders, "max_size_cycle")
        return combined_val_loaders