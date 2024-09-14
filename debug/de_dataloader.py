import logging
from pathlib import Path
import sys
from typing import List, Union
import os

import numpy as np
from tqdm import tqdm
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
        if x.ndim == 0:
            print(f'{prefix},{type(x)},shape={x.shape}, {x}')
        else:
            print(f'{prefix},{type(x)},shape={x.shape}')
    elif isinstance(x, bool) or isinstance(x, int):
        print(f'{prefix}:{type(x)},{x}')
    else:
        raise TypeError(f'Unexpected type {type(x)}')


def print_batch(prefix, x, depth=0):
    if isinstance(x, str) or isinstance(x, bool) or isinstance(x, int):
        print_leaf(prefix, x)
        return
    elif isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        print_leaf(prefix, x)
        return
    elif isinstance(x, dict):
        print(f'{prefix}: Dict,keys={x.keys()}')
        for k, v in x.items():
            if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                print_leaf(('-' * depth) + k, v)
            else:
                print_batch(('-' * depth) + k, v, depth + 1)
    elif isinstance(x, list):
        print(f'{prefix}: List,len={len(x)},elem:{type(x[0])}')
        if isinstance(x[0], torch.Tensor) or isinstance(x[0], np.ndarray):
            print_batch(('-' * depth) + '[0]', x[0], depth + 1)
    else:
        raise TypeError(f'type {type(x)} not supported. x must be torch.Tensor or list or dict')


@hydra.main(config_path="../conf", config_name="da_d_hk")
def main(cfg: DictConfig) -> None:
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    print('[DEBUG] datamodule loaded')
    datamodule.setup()

    for dataset_key, loader in datamodule.train_dataloader().items():
        print(('=' * 20) + f' Dataset {dataset_key} ' + ('=' * 20))
        print(f'Dataloader len={len(loader)}')
        for idx, example in enumerate(tqdm(loader)):
            if idx >= 20:
                continue
            # else:
            #     print_batch(f'Batch@{idx}th', example)
            # print(('-' * 20) + ' Batch End ' + ('-' * 20))

    '''
    Lang: Dict,keys=dict_keys(['robot_obs', 'rgb_obs', 'depth_obs', 'actions', 'state_info', 'use_for_aux_lang_loss', 'lang', 'lang_text', 'idx', 'future_frame_diff'])
    robot_obs,<class 'torch.Tensor'>,shape=torch.Size([16, 2, 8])            
    rgb_obs: Dict,keys=dict_keys(['rgb_static', 'rgb_gripper', 'gen_static', 'gen_gripper'])                                                                      
    -rgb_static,<class 'torch.Tensor'>,shape=torch.Size([16, 2, 3, 224, 224])
    -rgb_gripper,<class 'torch.Tensor'>,shape=torch.Size([16, 2, 3, 84, 84])                                                                                      
    -gen_static,<class 'torch.Tensor'>,shape=torch.Size([16, 1, 3, 112, 112])
    -gen_gripper,<class 'torch.Tensor'>,shape=torch.Size([16, 1, 3, 112, 112])
    depth_obs: Dict,keys=dict_keys([])                                       
    actions,<class 'torch.Tensor'>,shape=torch.Size([16, 10, 7])                                                                                                  
    state_info: Dict,keys=dict_keys(['robot_obs', 'scene_obs'])    
    -robot_obs,<class 'torch.Tensor'>,shape=torch.Size([16, 2, 15])                                                                                               
    -scene_obs,<class 'torch.Tensor'>,shape=torch.Size([16, 2, 24])          
    use_for_aux_lang_loss,<class 'torch.Tensor'>,shape=torch.Size([16])                                                                                           
    lang,<class 'torch.Tensor'>,shape=torch.Size([16, 1024])                 
    lang_text: List,len=16,elem:<class 'str'>                                                                                                                     
    idx,<class 'torch.Tensor'>,shape=torch.Size([16])                        
    future_frame_diff,<class 'torch.Tensor'>,shape=torch.Size([16])   
    
    Vis: Dict,keys=dict_keys(['robot_obs', 'rgb_obs', 'depth_obs', 'actions', 'state_info', 'lang', 'idx', 'future_frame_diff'])
    robot_obs,<class 'torch.Tensor'>,shape=torch.Size([16, 2, 8])
    rgb_obs: Dict,keys=dict_keys(['rgb_static', 'rgb_gripper', 'gen_static', 'gen_gripper'])
    -rgb_static,<class 'torch.Tensor'>,shape=torch.Size([16, 2, 3, 224, 224])
    -rgb_gripper,<class 'torch.Tensor'>,shape=torch.Size([16, 2, 3, 84, 84])
    -gen_static,<class 'torch.Tensor'>,shape=torch.Size([16, 1, 3, 112, 112])
    -gen_gripper,<class 'torch.Tensor'>,shape=torch.Size([16, 1, 3, 112, 112])
    depth_obs: Dict,keys=dict_keys([])
    actions,<class 'torch.Tensor'>,shape=torch.Size([16, 10, 7])
    state_info: Dict,keys=dict_keys(['robot_obs', 'scene_obs'])
    -robot_obs,<class 'torch.Tensor'>,shape=torch.Size([16, 2, 15])
    -scene_obs,<class 'torch.Tensor'>,shape=torch.Size([16, 2, 24])
    lang,<class 'torch.Tensor'>,shape=torch.Size([16, 0])
    idx,<class 'torch.Tensor'>,shape=torch.Size([16])
    future_frame_diff,<class 'torch.Tensor'>,shape=torch.Size([16])
    '''


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    os.environ["TOKENIZERS_PARALLELISM"] = 'True'
    os.environ["WANDB__SERVICE_WAIT"] = "300"  # alleviate ServiceStartTimeoutError
    main()
