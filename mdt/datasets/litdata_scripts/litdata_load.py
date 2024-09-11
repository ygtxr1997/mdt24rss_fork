import os
import time
import argparse
from typing import List
import numpy as np
import litdata as ld
from tqdm import tqdm

from debug.de_dataloader import print_batch


def vanilla_speed(data_path: str = '/home/geyuan/datasets/CALVIN/dataset/calvin_debug_dataset/training'):
    """ Speeed Debug for vanilla dataset """
    import torch
    from torch.utils.data import Dataset, DataLoader

    class DebugDataset(Dataset):
        def __init__(self, files: List[str]):
            super().__init__()
            self.files = files

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            return dict(np.load(self.files[idx], allow_pickle=True))

    all_files = os.listdir(data_path)
    npz_files = [os.path.join(data_path, x) for x in all_files if x.endswith('.npz') and 'episode' in x]
    print([os.path.basename(x) for x in npz_files])
    exit()
    bs = 128
    train_dataloader = torch.utils.data.DataLoader(
        DebugDataset(npz_files),
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=8,
    )
    batch_cnt = 0
    start_time = time.time()
    for idx, sample in enumerate(train_dataloader):
        # print_batch(f'Batch@{idx}', sample)
        batch_cnt += 1
    print(f'[litdata_load] Vanilla avg Speed: {batch_cnt / (time.time() - start_time)}batches/s, batch_size={bs}')


def litdata_speed(data_path: str = '/home/geyuan/datasets/CALVIN/dataset/litdata/calvin_debug_dataset/training'):
    """ Speed Debug for Litdata """
    train_dataset = ld.StreamingDataset(
        f'local:{data_path}',
        max_cache_size="10GB",
        shuffle=False,
        drop_last=False
    )
    bs = 128
    train_dataloader = ld.StreamingDataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=8,
    )

    batch_cnt = 0
    start_time = time.time()
    for idx, sample in enumerate(train_dataloader):
        # print_batch(f'Batch@{idx}', sample)
        batch_cnt += 1

    print(f'[litdata_load] Litdata avg Speed: {batch_cnt / (time.time() - start_time)}batches/s, batch_size={bs}')


def litdata_rand_read_speed(data_path: str = '/home/geyuan/datasets/CALVIN/dataset/litdata/task_D_D/training'):
    """ Speed Debug for Litdata """
    train_dataset = ld.StreamingDataset(
        f'local:{data_path}',
        max_cache_size="10GB",
        shuffle=False,
        drop_last=False
    )
    print(len(train_dataset))
    print(train_dataset.__len__())
    print_batch('[0]', train_dataset[0])
    print_batch('__getitem__(0)', train_dataset.__getitem__(0))
    exit()

    dict_ep_idx_to_dataset_order = np.load(os.path.join(data_path, "dict_ep_idx_to_dataset_order.npy"),
            allow_pickle=True).reshape(-1)[0]
    print('keys:', min(dict_ep_idx_to_dataset_order.keys()), max(dict_ep_idx_to_dataset_order.keys()))
    print('values:', min(dict_ep_idx_to_dataset_order.values()), max(dict_ep_idx_to_dataset_order.values()))

    bs = 128
    batch_cnt = 0
    start_time = time.time()
    for idx in range(len(train_dataset) // bs):
        batch_indices: List = np.random.randint(len(train_dataset), size=bs).tolist()
        for _ in range(bs):
            batch_data = train_dataset[batch_indices[_]]
        batch_cnt += 1

    print(f'[litdata_load] Litdata Rand-Read Speed: {batch_cnt / (time.time() - start_time)}batches/s, batch_size={bs}')


if __name__ == '__main__':
    # vanilla_speed()
    # litdata_speed()
    litdata_rand_read_speed()
