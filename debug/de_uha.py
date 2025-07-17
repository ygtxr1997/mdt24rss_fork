# minimum working example to load the OXE dataset
import os
import time
import warnings

import hydra
import numpy as np
import mediapy
from PIL import Image
from tqdm import tqdm
from uha import make_pytorch_oxe_iterable_dataset, get_octo_dataset_tensorflow, get_single_dataset_tensorflow, multi_worker_iterable_dataset
from omegaconf import DictConfig, OmegaConf
import torch


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
        warnings.warn(f'type {type(x)} not supported. x must be torch.Tensor or list or dict')
        print(f'{prefix}: {type(x)}')


@hydra.main(config_path="../conf", config_name="config_oxe_hk.yaml")
def main(cfg: DictConfig):
    cfg = cfg.datamodule  # get `datamodule`
    if "HOME" in cfg:
        os.environ["HOME"] = cfg.HOME
    # load Training Dataset from TensorflowDatasets
    # dataset = get_octo_dataset_tensorflow(cfg, train=True)
    dataset = get_single_dataset_tensorflow(cfg, train=True)  #.repeat().unbatch()
    is_single_dataset = True
    batch_size = 4
    del cfg.interleaved_dataset_cfg.frame_transform_kwargs.image_augment_kwargs  # comment out to get augmented data
    cfg_transforms = OmegaConf.to_object(cfg.transforms)
    language_encoder = hydra.utils.instantiate(cfg.language_encoders)
    # create Pytorch Train Dataset
    # dataloader = make_pytorch_oxe_iterable_dataset(
    #     dataset, train=True, batch_size=batch_size, transform_dict=cfg_transforms, num_workers=0, pin_memory=True,
    #     language_encoder=language_encoder, is_single_dataset=is_single_dataset, main_process=True)
    # dataloader = multi_worker_iterable_dataset(dataset, train=True, batch_size=batch_size, transform_dict=cfg_transforms, num_workers=0, pin_memory=True, language_encoder=language_encoder, is_single_dataset=is_single_dataset, main_process=True)
    dataloader = make_pytorch_oxe_iterable_dataset(
        dataset, train=True, batch_size=batch_size, transform_dict=cfg_transforms, language_encoder=language_encoder,
        is_single_dataset=is_single_dataset,
        num_workers=0,
        pin_memory=True,
        main_process=True,
    )
    val_dataloader = make_pytorch_oxe_iterable_dataset(
        dataset, train=False, batch_size=batch_size, transform_dict=cfg_transforms, language_encoder=language_encoder,
        is_single_dataset=is_single_dataset,
        num_workers=0,
        pin_memory=True,
        main_process=True,
    )
    generator = iter(dataloader)
    generator = iter(val_dataloader)
    time.sleep(1)

    # for sample in dataloader:
    for step in tqdm(range(50)):
        sample = next(generator)
        # print("Top-level keys: ", sample.keys())
        # print("rgb_obs keys: ", sample["rgb_obs"].keys())
        # print("Task keys: ", sample["task"].keys())
        # print("action: ", sample["action"][:, 0, -1])
        print(step)
        print_batch("sample", sample)

        encoding = sample['task']['language_instruction']
        input_ids = encoding["input_ids"]
        print(input_ids.shape)
        decoded_texts = language_encoder.tokenizer.batch_decode(input_ids.squeeze(1), skip_special_tokens=True)
        print(decoded_texts)

        image_primary = sample['observation']['image_primary'].float()
        print(type(image_primary))
        print(image_primary.mean(), image_primary.max(), image_primary.min(),)

        act1 = sample['action'][:, :, :, 0]
        print(act1.mean(), act1.max(), act1.min())
        act1 = sample['action'][:, :, :, 3]
        print(act1.mean(), act1.max(), act1.min())
        act1 = sample['action'][:, :, :, 6]
        print(act1.mean(), act1.max(), act1.min())

        images, image_secondary, image_wrist = [], [], []
        for i in range(batch_size):
            Image.fromarray(sample['observation']['image_primary'][i, 0].numpy()).save(f"tmp_primary_{i:02d}.png")
            images.append(
                sample["observation"]["image_primary"][i, 0].numpy())  # [batch_size, window_size, rgb, width, height]
            image_secondary.append(
                sample["observation"]["image_secondary"][i, 0].numpy())  # [batch_size, window_size, rgb, width, height]
        mediapy.write_video("tmp_primary.mp4", images, fps=10, qp=18)
        mediapy.write_video("tmp_secondary.mp4", image_secondary, fps=10, qp=18)

        # print("Top-level keys: ", sample.keys())
        # print("Task keys: ", sample["task"].keys())
        # print("task image_primary shape: ", sample["task"]["image_primary"].shape)
        # print("observation image_primary shape: ", sample["observation"]["image_primary"].shape)
        # print("task language tokens: ", sample["task"]["language_instruction"]["input_ids"].shape)
        # print("observation_proprio shape: ", sample["observation"]["proprio"].shape)
        break


if __name__ == "__main__":
    use_ddp_env = True
    if use_ddp_env:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
        print(torch.cuda.is_available())
        print(torch.cuda.device_count())
        os.environ["TOKENIZERS_PARALLELISM"] = 'True'
        os.environ["WANDB__SERVICE_WAIT"] = "300"  # alleviate ServiceStartTimeoutError
        main()
