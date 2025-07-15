import copy
import os
import numpy as np
import torch
import torch.utils.data
from torchvision.transforms import transforms

from robokit.data.tcl_datasets import TCLDataset, TCLDatasetHDF5


class TCLImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 # RoboKit Dataset
                 data_root: str,
                 # Data sequence
                 horizon: int,
                 pad_before: int,
                 pad_after: int,
                 # Data format
                 shape_meta: dict,
                 # MDT related
                 batch_size: int = 64,
                 num_workers: int = 8,
                 key: str = "lang",
                 chose_ratio: float = 1.,
                 img_gen_frame_diff: int = 3,
                 # Others
                 seed: int = 42,
                 val_ratio: float = 0.01,
                 split: str = "train",
                 remap_index: np.ndarray = None,  # only for val_set
                 max_train_episodes: int = 90,
                 transform_color_jitter: bool = True,
                 # RoboKit Dataset
                 h5_path: str = None,
                 use_h5: bool = False,
                 ):
        # RoboKit Dataset
        self.data_root = data_root
        self.shape_meta = shape_meta
        self.load_keys = ["rel_actions", "primary_rgb", "gripper_rgb", "robot_obs", "language_text"]
        self.h5_path = h5_path
        self.use_h5 = use_h5
        if not use_h5:
            self.tcl_dataset = TCLDataset(data_root, use_extracted=True, load_keys=self.load_keys)
        else:
            assert os.path.exists(h5_path), f"h5_path: ${h5_path} not exists"
            self.tcl_dataset = TCLDatasetHDF5(
                data_root, h5_path,
                use_extracted=True, load_keys=self.load_keys)
        self.data_meta = self.tcl_dataset.load_statistics_from_json(os.path.join(data_root, "statistics.json"))
        self.all_rel_actions = self.tcl_dataset.extracted_data["rel_actions"]
        self.dataset_min = np.array(self.data_meta["min"])
        self.dataset_max = np.array(self.data_meta["max"])
        self.dataset_mean = np.array(self.data_meta["mean"])
        self.dataset_std = np.array(self.data_meta["std"])
        self.dataset_total_len = self.data_meta["total_len"]

        self.tasks = self.tcl_dataset.tasks
        self.task_lengths = self.tcl_dataset.task_lengths
        self.ep_fns = self.tcl_dataset.ep_fns
        self.map_index_to_task_id = self.tcl_dataset.map_index_to_task_id

        # MDT related
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chose_ratio = chose_ratio
        self.img_gen_frame_diff = img_gen_frame_diff

        # Others
        self.seed = seed
        self.val_ratio = val_ratio
        self.split = split
        self.max_train_episodes = max_train_episodes

        # 创建重映射索引
        if self.split == "train":
            self.index_all = np.arange(len(self.tcl_dataset))  # split train and val
            np.random.shuffle(self.index_all)
            val_size = int(len(self.index_all) * val_ratio)
            self.index_val = self.index_all[:val_size]
            self.index_train = self.index_all[val_size:]
            self.remap_index = self.index_train  # local to global
        elif self.split == "val":
            self.remap_index = remap_index  # provided by the train_set
        else:
            raise NotImplementedError("split not implemented")

        # Sampling a data sequence
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.task_prefix_lengths = [0 for _ in range(len(self.task_lengths))]
        for i in range(1, len(self.task_prefix_lengths)):  # 3 means [0+1+2]
            self.task_prefix_lengths[i] = self.task_prefix_lengths[i - 1] + self.task_lengths[i - 1]

        # Data format and preprocessing
        self.obs_image_shape = shape_meta["obs"]["image"]["shape"]  # [3, H, W]
        self.obs_gripper_shape = shape_meta["obs"]["gripper"]["shape"] if "gripper" in shape_meta["obs"] else None
        self.joint_state_shape = shape_meta["obs"]["joint_state"]["shape"]
        self.action_shape = shape_meta["action"]["shape"]  # [7,]
        obs_image_wh_ratio = float(self.obs_image_shape[2]) / float(self.obs_image_shape[1])  # wh 4:3=16:12=12:9
        transform_list = [
            transforms.ToPILImage(),  # wh 16:9
            transforms.Resize(self.obs_image_shape[1:]),
        ]
        if transform_color_jitter:
            # DP used
            # transforms.RandomResizedCrop(size=self.obs_image_shape[1:], scale=(0.68, 0.82),  # 12/16=0.75
            #                              ratio=(0.9 * obs_image_wh_ratio, 1.1 * obs_image_wh_ratio)),  # not using this would be better?
            # transform_list.append(transforms.ColorJitter(brightness=0.05,
            #                                              contrast=0.05,
            #                                              saturation=0.05,
            #                                              hue=0.05))
            # Suggested by GPT4.1
            # trans_crop = transforms.RandomResizedCrop(
            #     size=self.obs_image_shape[1:],
            #     scale=(0.8, 1.0),  # 从50%到100%都可能 crop
            #     ratio=(0.9, 1.1)
            # )  # Shut down crop
            # transform_list.insert(1, trans_crop)
            transform_list.append(transforms.ColorJitter(brightness=0.4,
                                                         contrast=0.4,
                                                         saturation=0.4,
                                                         hue=0.15))
        transform_list.append(transforms.ToTensor())
        self.obs_image_transform = transforms.Compose(transform_list)  # Similar augmentation params with OCTO

        gripper_transform_list = copy.deepcopy(transform_list)
        gripper_transform_list[1] = transforms.Resize(self.obs_gripper_shape[1:])
        self.obs_gripper_transform = transforms.Compose(gripper_transform_list)

        gen_transform_list = copy.deepcopy(gripper_transform_list)
        gen_transform_list[1] = transforms.Resize((112, 112))
        self.gen_transform = transforms.Compose(gen_transform_list)

        print(f"[TCLImageDataset] dataset loaded, split={self.split}, val_ratio={self.val_ratio}, len={len(self)}; "
              f"total_len={self.dataset_total_len}, "
              f"action_min={self.dataset_min:.6f}, action_max={self.dataset_max:.6f}, "
              f"action_mean={self.dataset_mean:.6f}, action_std={self.dataset_std:.6f}")

    def get_validation_dataset(self):
        return self.create_val_dataset(self)

    @classmethod
    def create_val_dataset(cls, instance: 'TCLImageDataset'):
        val_set = cls(
            data_root=instance.data_root,
            horizon=instance.horizon,
            pad_before=instance.pad_before,
            pad_after=instance.pad_after,
            shape_meta=instance.shape_meta,
            seed=instance.seed,
            val_ratio=instance.val_ratio,
            split='val',
            remap_index=instance.index_val,
            max_train_episodes=instance.max_train_episodes,
            use_h5=False,  # no need to use h5
            batch_size=16,
        )
        val_set.tcl_dataset.total_length = min(6000, val_set.tcl_dataset.total_length)
        return val_set

    # def get_normalizer(self, **kwargs) -> LinearNormalizer:
    #     normalizer = LinearNormalizer()
    #     obs_keys = self.shape_meta["obs"].keys()
    #     for obs_key in obs_keys:
    #         normalizer[obs_key] = EmptyNormalizer.create_identity()
    #     # normalizer['image'] = EmptyNormalizer.create_identity()
    #     # normalizer['joint_state'] = EmptyNormalizer.create_identity()
    #     normalizer['action'] = EmptyNormalizer.create_identity()
    #     return normalizer

    def __len__(self):
        return len(self.remap_index)

    def __getitem__(self, abs_idx):
        abs_idx = self.remap_index[abs_idx]  # convert relative index to global absolute index

        abs_idx = abs_idx % self.__len__()
        obs_data = self._get_obs_data(abs_idx)
        act_data = self._get_act_data(abs_idx)
        # item_data = {
        #     "obs": obs_data,
        #     "action": act_data,
        # }
        item_data = {
            "robot_obs": obs_data["joint_state"],  # (T,6)
            "rgb_obs": {
                "rgb_static": obs_data['image'],
                "rgb_gripper": obs_data['gripper'],  # (T,C,H,W)
                "gen_static": obs_data['gen_primary'],  # (1,C,H,W)
                "gen_gripper": obs_data['gen_gripper'],
            },
            "depth_obs": {},
            "actions": act_data,  # (T,7)
            "state_info": {
                "scene_obs": np.zeros((1, 24)),
                "robot_obs": np.zeros((1, 15))
            },
            "lang": {},
            "lang_text": obs_data['language'],
            "idx": abs_idx,
            "future_frame_diff": self.img_gen_frame_diff,
        }
        return item_data

    def _abs_idx_to_rel_idx(self, abs_idx: int):
        task_id = self.map_index_to_task_id[abs_idx]
        task_len = self.task_lengths[task_id]
        task_prefix_len = self.task_prefix_lengths[task_id]
        rel_idx = abs_idx - task_prefix_len
        assert 0 <= rel_idx < task_len
        return rel_idx, task_id

    def _rel_idx_to_abs_idx(self, rel_idx: int, task_id: int):
        task_prefix_len = self.task_prefix_lengths[task_id]
        task_len = self.task_lengths[task_id]
        if rel_idx < 0:
            abs_idx = None
        elif rel_idx >= task_len:
            abs_idx = None
        else:
            abs_idx = rel_idx + task_prefix_len
            assert task_prefix_len <= abs_idx < task_prefix_len + task_len
        return abs_idx

    def _get_abs_obs_indices(self, abs_now_idx: int):
        # [start_idx, end_idx)
        rel_now_idx, task_id = self._abs_idx_to_rel_idx(abs_now_idx)
        start_idx = rel_now_idx - self.pad_before
        end_idx = rel_now_idx + 1
        rel_indices = list(range(start_idx, end_idx))
        return [self._rel_idx_to_abs_idx(rel_id, task_id) for rel_id in rel_indices]

    def _get_abs_gen_index(self, abs_now_idx: int):
        # [start_idx, end_idx)
        rel_now_idx, task_id = self._abs_idx_to_rel_idx(abs_now_idx)
        gen_idx = rel_now_idx + self.img_gen_frame_diff
        abs_gen_idx = self._rel_idx_to_abs_idx(gen_idx, task_id)
        return abs_gen_idx if abs_gen_idx is not None else abs_now_idx

    def _get_abs_act_indices(self, abs_now_idx: int):
        # [start_idx, end_idx)
        rel_now_idx, task_id = self._abs_idx_to_rel_idx(abs_now_idx)
        start_idx = rel_now_idx
        end_idx = rel_now_idx + self.horizon
        rel_indices = list(range(start_idx, end_idx))
        return [self._rel_idx_to_abs_idx(rel_id, task_id) for rel_id in rel_indices]

    def _get_obs_data(self, abs_idx: int):
        task_id = self.map_index_to_task_id[abs_idx]
        abs_obs_indices = self._get_abs_obs_indices(abs_idx)
        # print(f"[DEBUG] _get_obs_data: abs_idx={abs_idx}, task_id={task_id}, "
        #       f"task_1st_ep={self.task_prefix_lengths[task_id]}, "
        #       f"task_last_ep={self.task_prefix_lengths[task_id] + self.task_lengths[task_id]}"
        #       )
        # print(f"[DEBUG] _get_obs_data: abs_obs_indices={abs_obs_indices} ")
        abs_gen_index = self._get_abs_gen_index(abs_idx)
        obs_keys = self.shape_meta["obs"].keys()
        obs_data = {
            k: [] for k in obs_keys
        }
        language = ""
        for idx in abs_obs_indices:
            if idx is None:
                c, h, w = self.obs_image_shape
                zero_rgb = torch.ones((c, h, w)).to(torch.float32) * -1  # all -1
                primary_rgb = zero_rgb
                gripper_rgb = zero_rgb
                tcp_pose = torch.zeros((6,)).to(torch.float32)
            else:
                sample_dict = self.tcl_dataset.__getitem__(idx)
                primary_rgb = sample_dict['primary_rgb']  # (H,W,C)
                gripper_rgb = sample_dict['gripper_rgb']  # (H,W,C)
                tcp_pose = joint_state = sample_dict['robot_obs'][:6]  # (6,)
                language = sample_dict['language_text']  # string
                # Preprocess
                primary_rgb = self.obs_image_transform(primary_rgb)  # (C,H,W), in [0, 1]
                primary_rgb = primary_rgb * 2. - 1.  # in [-1, 1]
                tcp_pose = torch.from_numpy(np.concatenate([tcp_pose, np.zeros(2)])).to(torch.float32)  # (8,)
                if "gripper" in obs_keys:
                    gripper_rgb = self.obs_gripper_transform(gripper_rgb)
                    gripper_rgb = gripper_rgb * 2. - 1.
            obs_data["image"].append(primary_rgb)
            obs_data["joint_state"].append(tcp_pose)
            if "gripper" in obs_keys:
                obs_data["gripper"].append(gripper_rgb)
        obs_data = {k: torch.stack(v) for k, v in obs_data.items()}
        # obs_data["image"] = torch.stack(obs_data["image"])  # should be (T,C,H,W)
        # obs_data["joint_state"] = torch.stack(obs_data["joint_state"])  # (T,6)
        obs_data['language'] = language

        # Get goal data for generation
        goal_dict = self.tcl_dataset.__getitem__(abs_gen_index)
        gen_primary = self.gen_transform(goal_dict['primary_rgb'])  # (C,H,W,), in [0,1]
        gen_gripper = self.gen_transform(goal_dict['gripper_rgb'])
        gen_primary = gen_primary * 2. - 1.
        gen_gripper = gen_gripper * 2. - 1.
        obs_data['gen_primary'] = gen_primary[None, :, :, :]
        obs_data['gen_gripper'] = gen_gripper[None, :, :, :]

        return obs_data

    def _get_act_data(self, abs_idx: int):
        task_id = self.map_index_to_task_id[abs_idx]
        abs_act_indices = self._get_abs_act_indices(abs_idx)
        # print(f"[DEBUG] _get_act_data: abs_idx={abs_idx}, task_id={task_id}, "
        #       f"task_1st_ep={self.task_prefix_lengths[task_id]}, "
        #       f"task_last_ep={self.task_prefix_lengths[task_id] + self.task_lengths[task_id]}"
        #       )
        # print(f"[DEBUG] _get_obs_data: abs_obs_indices={abs_act_indices} ")
        act_data = []
        for idx in abs_act_indices:
            if idx is None:
                zero_act = np.zeros(self.action_shape).astype(np.float32)
                act_data.append(zero_act)
            else:
                rel_action = self.all_rel_actions[idx]  # (7,)
                act_data.append(rel_action)
        act_data = np.stack(act_data)  # (T,7), in [act_min, act_max]
        act_data = (act_data - self.dataset_min) / (self.dataset_max - self.dataset_min)  # norm here, in [0,1]
        return act_data * 2. - 1.  # in [-1,1]
