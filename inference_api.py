import base64
import io
import os
from typing import List, Dict, Union
from functools import lru_cache
from pathlib import Path
import json
import pydantic
import copy

import hydra
import torch
import numpy as np
from fastapi import FastAPI
from omegaconf import OmegaConf
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms as T
from optree import tree_map

from mdt.models.mdt_agent import MDTAgentOXE


""" How to use me?
$ CUDA_VISIBLE_DEVICES=9 uvicorn inference_api:app --port 6060
"""
app = FastAPI()
max_cache_action = 7  # will be sent to the evaluator through network

log_time = "2025-07-16/16-24-39"  # pot source
w_idx = -1
dataset_name = "pepper"  # shovel; pot, pot_light; pepper; coffee


class StepRequestWithObservation(pydantic.BaseModel):
    primary_rgb: List[str]
    gripper_rgb: List[str]
    instruction: str
    joint_state: List[List[float]]


def load_model_from_safetensor(
    filedir: Path,
    overwrite_cfg: dict = {},
):
    # if not filedir.is_dir():
    #     raise ValueError(f"not valid file path: {str(filedir)}")

    # Traverse the directory to find all image files within two-level subdirectories
    config_path = None
    statistics_path = None  # for action denormalize
    ckpt_paths = []
    for root, dirs, files in os.walk(filedir):
        for file in files:
            if ".hydra" in root and file == "config.yaml":
                config_path = os.path.join(root, file)
            elif "checkpoints" in root and file == "statistics.json":
                statistics_path = os.path.join(root, file)
            elif file.endswith(".ckpt"):
                ckpt_paths.append(os.path.join(root, file))

    def move_latest_to_end(ckpt_paths):
        # 分别提取不含latest和含latest的元素
        non_latest = [path for path in ckpt_paths if 'latest' not in path]
        latest = [path for path in ckpt_paths if 'latest' in path]
        # 合并，latest元素放到最后
        return non_latest + latest

    ckpt_paths.sort()
    ckpt_paths = move_latest_to_end(ckpt_paths)
    print(f"Find ckpt: {[(os.path.basename(x)) for x in ckpt_paths]}")
    config = OmegaConf.load(str(config_path))
    ckpt_path = ckpt_paths[w_idx]
    assert config_path is not None and ckpt_path is not None, "config or ckpt file not found!"
    print(f"config: {config_path}; ckpt: {ckpt_path}; statistics: {statistics_path}")

    print(f"Loading model from {ckpt_path}")
    load_cfg = OmegaConf.create({**OmegaConf.to_object(config.model), **{"optimizer": None}, **overwrite_cfg})
    # load_cfg["ckpt_path"] = str(ckpt_path)

    model = hydra.utils.instantiate(load_cfg)
    weights = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(weights['state_dict'], strict=False)

    # 3. Meta data statistics
    ## Try to load statistics.json from original dataset dir
    if statistics_path is None:
        if "2025.05.11" in log_time or "2025.05.13" in log_time:
            dataset_dir = "collected_data_0507"
        elif dataset_name == "shovel":
            dataset_dir = "collected_data_0514_shovel_source"
        elif dataset_name == "pot":
            dataset_dir = "0627_pot_source"
        elif dataset_name == "pot_light":
            dataset_dir = "0627_pot_light"
        elif dataset_name == "pot_object":
            dataset_dir = "0627_pot_object"
        elif dataset_name == "pepper":
            dataset_dir = "0704_pepper_source"
        elif dataset_name == "coffee":
            dataset_dir = "0709_coffee_source"
        else:
            raise KeyError("statistics.json not found!")

        statistics_path = f"/home/geyuan/local_soft/TCL/{dataset_dir}/statistics.json"

    # with open(f"/home/geyuan/local_soft/TCL/{dataset_dir}/statistics.json", 'r') as json_file:
    #     statistics = json.load(json_file)
    #     data_min = torch.from_numpy(np.array(statistics['min']))
    #     data_max = torch.from_numpy(np.array(statistics['max']))

    with open(statistics_path, "r") as json_file:
        statistics = json.load(json_file)
        statistics = {
            "min": np.array(statistics["min"]),
            "max": np.array(statistics["max"]),
            "mean": np.array(statistics["mean"]),
            "std": np.array(statistics["std"]),
            "total_len": np.array(statistics["total_len"]),
        }

    # 4. Other settings
    model.rollout_step_counter = 0
    model.multistep = max_cache_action

    print(f"[Info] Finished loading model {ckpt_path}, statistics: {statistics_path}")
    return {
        'model': model,
        'config': config,
        'ckpt_path': ckpt_path,
        'statistics': statistics,
    }


@lru_cache
def get_agent(device: str) -> Dict:
    # Op1. Base OXE model
    # run_dir = "/home/geyuan/code/mdt24rss_fork/logs/runs/2025-03-01/19-16-36/"
    # Op2. HDFree model
    # run_dir = "/home/geyuan/code/mdt24rss_fork/logs/runs/2025-03-05/15-43-27/"
    # Op3. Base TCL model
    run_dir = f"/home/geyuan/code/mdt24rss_fork/logs/runs/{log_time}/"

    loaded_model_and_others = load_model_from_safetensor(
        Path(run_dir),
    )
    model = loaded_model_and_others['model']
    model.eval()
    model.freeze()
    model.to(device)

    return {
        'model': model,
        'config': loaded_model_and_others['config'],
        'ckpt_path': loaded_model_and_others['ckpt_path'],
        'statistics': loaded_model_and_others['statistics'],
    }


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.get("/reset")
def model_reset():
    agent = get_agent("cuda")['model']
    agent.reset()
    return {"max_cache_action": max_cache_action}


@app.post("/step")
def model_step(step_request: StepRequestWithObservation):
    """ How to use this?
    pip install uvicorn fastapi
    CUDA_VISIBLE_DEVICES=9 uvicorn inference_api:app --port 22780
    """
    agent_and_others = get_agent("cuda")
    agent = agent_and_others['model']
    hydra_config = agent_and_others['config']
    ckpt_path = agent_and_others['ckpt_path']
    statistics: Dict[str, np.ndarray] = agent_and_others['statistics']
    print("Using cached ckpt from:", ckpt_path, ". model type:", type(agent))

    # 1. Decode observation from received request
    shape_meta = hydra_config.datamodule.datasets.lang_dataset.shape_meta
    image_shape = shape_meta["obs"]["image"]["shape"]  # (3,224,224)
    gripper_shape = shape_meta["obs"]["gripper"]["shape"]  # (3,84,84)
    # gripper_shape = (3, 224, 224)
    print("[DEBUG] image:", image_shape, "gripper:", gripper_shape)

    instruction_text = step_request.instruction
    joint_state = step_request.joint_state  # List:(T,6)
    joint_state = torch.from_numpy(
        np.concatenate([np.array(joint_state),
                        np.zeros((len(joint_state), 2))], axis=1)
    ).to("cuda").unsqueeze(0)  # (B,T,8)
    obs_dict = {
        "robot_obs": joint_state,  # should be (B,T,8)
        "rgb_obs": {
            "rgb_static": None,
            "rgb_gripper": None,
        },
    }
    goal_dict = {
        "lang_text": instruction_text,
    }

    if True:  # always enter
        primary_imgs = []
        for idx, primary_img in enumerate(step_request.primary_rgb):
            primary_img = base64.b64decode(primary_img)
            primary_img = Image.open(io.BytesIO(primary_img), formats=["JPEG"])
            primary_img.save(f"tmp_primary_{idx}.jpg")

            rgb_transform = T.Compose([
                T.Resize(image_shape[1:]),
                T.ToTensor(),
            ])
            primary_img = rgb_transform(primary_img)  # (C,H,W), in [0,1]
            primary_img = primary_img * 2. - 1.  # in [-1,1]
            primary_imgs.append(primary_img)

        gripper_imgs = []
        for idx, gripper_img in enumerate(step_request.gripper_rgb):
            gripper_img = base64.b64decode(gripper_img)
            gripper_img = Image.open(io.BytesIO(gripper_img), formats=["JPEG"])
            gripper_img.save(f"tmp_gripper_{idx}.jpg")

            rgb_transform = T.Compose([
                T.Resize(gripper_shape[1:]),
                T.ToTensor(),
            ])
            gripper_img = rgb_transform(gripper_img)  # (C,H,W), in [0,1]
            gripper_img = gripper_img * 2. - 1.  # in [-1,1]
            gripper_imgs.append(gripper_img)

        # 2. Preprocess, e.g resize, normalize, to_tensor, to_device
        primary_img = torch.stack(primary_imgs, dim=0)  # (T,C,H,W)
        primary_img = primary_img.to("cuda").unsqueeze(0)  # (B,T,C,H,W)
        gripper_img = torch.stack(gripper_imgs, dim=0)
        gripper_img = gripper_img.to("cuda").unsqueeze(0)

        obs_dict["rgb_obs"]["rgb_static"] = primary_img  # (B,T,C,H,W)
        obs_dict["rgb_obs"]["rgb_gripper"] = gripper_img

    # 3.a Model inference (we have controlled the communication frequency in the evaluator)
    # action_idx = agent.infer_frame_idx % max_cache_action
    with torch.no_grad():
        pred_action_seq = agent(
            obs=obs_dict,
            goal=goal_dict,
        )  # (B,T,7)
        # print("pred_action:", pred_action_seq.shape)
        action = copy.deepcopy(pred_action_seq[0, :])  # remove batch_dim, (T,7)
        agent.pred_action_seq = pred_action_seq

    # 3.b Postprocess
    # Denorm helper
    def denorm_action(normed_action: np.ndarray, norm_action_type: str, meta_data: Dict[str, np.ndarray]):
        # Consider different norm types
        if norm_action_type == "minmax" or norm_action_type is None:
            dataset_min = meta_data['min']
            dataset_max = meta_data['max']

            # 从 [-1,1] 转换到 [0,1]
            action_01 = (normed_action + 1.) / 2.

            # 从 [0,1] 转换回原始范围
            denormed_action = action_01 * (dataset_max - dataset_min) + dataset_min

        elif norm_action_type == "mean":
            dataset_min = meta_data['min']
            dataset_max = meta_data['max']
            dataset_mean = meta_data['mean']
            dataset_std = meta_data['std']

            # 分割为姿态和抓手数据
            pose_normed = normed_action[..., :-1]  # (T, 6)
            gripper_normed = normed_action[..., -1:]  # (T, 1)

            # 反归一化姿态数据 (从标准化恢复)
            pose_denormed = pose_normed * dataset_std[:-1] + dataset_mean[:-1]

            # 反归一化抓手数据 (从 [-1,1] 恢复到原始范围)
            gripper_01 = (gripper_normed + 1.) / 2.  # 转换到 [0,1]
            gripper_denormed = gripper_01 * (dataset_max[-1:] - dataset_min[-1:]) + dataset_min[-1:]

            # 重新拼接
            denormed_action = np.concatenate([pose_denormed, gripper_denormed], axis=-1)

        else:
            assert norm_action_type == "identity", "norm_action_type not supported"
            denormed_action = normed_action.copy()

        return denormed_action

    # if norm_type is None or norm_type == "minmax":
    #     action = (action * 0.5 + 0.5).cpu()  # in [0,1], denormalize
    #     action = action * (data_max - data_min) + data_min
    # elif norm_type == "mean":
    #     pass
    # else:
    #     assert norm_type == "identity"
    #     action = action.cpu()

    norm_type = hydra_config.get("norm_action_type", None)
    action = denorm_action(action.cpu().numpy(), norm_action_type=norm_type, meta_data=statistics)
    print("tcp_pose/joint_state:", joint_state.shape, "; lang:", instruction_text[:20], "; norm_type:", norm_type)

    # 4. Return results
    cache_action = copy.deepcopy(agent.pred_action_seq[0])  # remove batch dim
    # cache_action = (cache_action * 0.5 + 0.5).cpu()  # in [0,1]
    # cache_action = cache_action * (data_max - data_min) + data_min
    cache_action = denorm_action(cache_action.cpu().numpy(), norm_type, statistics)
    for act_idx in range(cache_action.shape[0]):
        if cache_action[act_idx, 6:] >= 0.5:
            cache_action[act_idx, 6:] = 1
        else:
            cache_action[act_idx, 6:] = 0

    return {"action": cache_action.tolist()}


def main_oxe():
    # Create a white image
    image = Image.new("RGB", (224, 224), (255, 255, 255))

    # Save to io.Bytes with JPEG format
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes = image_bytes.getvalue()

    # Encode to base64
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    model_reset()

    res = model_step(
        StepRequest(
            image=image_base64,
            instr="Move the box to the left",
        )
    )
    print(res)


def main_tcl():
    from robokit.service.service_connector import ServiceConnector
    agent = get_agent("cuda")
    zero_rgb = np.zeros((1, 480, 848, 3), dtype=np.uint8)  # (T,H,W,C)
    debug_request = StepRequestWithObservation(
        primary_rgb=ServiceConnector.img_np_to_base64(zero_rgb),
        gripper_rgb=ServiceConnector.img_np_to_base64(zero_rgb),
        instruction="none",
        joint_state=[[0.] * 6] * 1,
    )
    pred_action = model_step(debug_request)
    print("Output:", pred_action)


if __name__ == "__main__":
    main_tcl()
