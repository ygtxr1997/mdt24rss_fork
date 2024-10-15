import logging
import os
from typing import Any, Dict, NamedTuple, Optional, Tuple
from functools import partial
import copy

import torch
import torch.nn as nn
from litdata.processing.utilities import catch
from torch.nn import functional as F
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch.distributed as dist
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
import einops
import torch.optim as optim
import wandb

from mdt.models.edm_diffusion.gc_sampling import *
from mdt.models.edm_diffusion.utils import append_dims
from mdt.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler
from mdt.models.perceptual_encoders.no_encoder import NoEncoder
from mdt.models.networks.transformers.transformer_blocks import ClipStyleProjection
from mdt.callbacks.ema import EMA
from mdt.models.perceptual_encoders.resnets import BesoResNetEncoder

logger = logging.getLogger(__name__)


def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

    for name, submodule in model.named_modules():
        # Adjusting the condition to capture the desired layers
        if '.' not in name or name.count('.') <= 10:  # Can be adjusted based on your model structure
            # Counting parameters including submodules
            submodule_params = sum(p.numel() for p in submodule.parameters())
            if submodule_params > 0:
                print(f"{name} - Total Params: {submodule_params}")


class MDTDomainAdaptVisualEncoder(pl.LightningModule):
    """
    The lightning module used for training.
    """

    def __init__(
            self,
            language_goal: DictConfig,
            visual_goal: DictConfig,
            img_gen: DictConfig,
            model: DictConfig,
            domain_adapt: DictConfig,
            optimizer: DictConfig,
            lr_scheduler: DictConfig,
            latent_dim: int = 512,
            multistep: int = 10,
            sampler_type: str = 'ddim',
            num_sampling_steps: int = 10,
            sigma_data: float = 0.5,
            sigma_min: float = 0.001,
            sigma_max: float = 80,
            noise_scheduler: str = 'exponential',
            sigma_sample_density_type: str = 'loglogistic',
            use_lr_scheduler: bool = True,
            act_window_size: int = 10,
            cont_alpha: int = 1,
            masked_beta: float = 1,
            use_distributed_clip: bool = False,
            use_text_not_embedding: bool = False,
            ckpt_path=None,
            seed: int = 42,
    ):
        super(MDTDomainAdaptVisualEncoder, self).__init__()
        self.automatic_optimization = False  # manually backward
        print('[MDTDomainAdaptVisualEncoder] Set automatic optimization to False!')
        self.latent_dim = latent_dim
        img_gen['context_dim'] = self.latent_dim
        self.static_resnet = BesoResNetEncoder(self.latent_dim)
        self.gripper_resnet = BesoResNetEncoder(self.latent_dim)
        self.act_window_size = act_window_size
        self.gen_img = hydra.utils.instantiate(img_gen).to(self.device)
        self.seed = seed
        self.use_lr_scheduler = use_lr_scheduler
        # goal encoders
        self.visual_goal = hydra.utils.instantiate(visual_goal).to(self.device)
        self.language_goal = hydra.utils.instantiate(language_goal) if language_goal else None
        # policy network
        self.model = hydra.utils.instantiate(model).to(self.device)
        self.modality_scope = "vis"
        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_hyperparameters()
        self.masked_beta = masked_beta
        # diffusion stuff
        self.sampler_type = sampler_type
        self.num_sampling_steps = num_sampling_steps
        self.noise_scheduler = noise_scheduler
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_sample_density_type = sigma_sample_density_type
        # for inference
        self.rollout_step_counter = 0
        self.multistep = multistep
        self.latent_goal = None
        self.plan = None
        self.state_recons = False
        self.cont_alpha = cont_alpha
        self.use_text_not_embedding = use_text_not_embedding
        # print_model_parameters(self.perceptual_encoder.perceiver_resampler)
        # for clip loss ground truth plot
        self.cont_loss = self.clip_auxiliary_loss
        self.cont_loss_type = 'infonce'
        self.use_distributed_clip = use_distributed_clip
        self.clip_proj = ClipStyleProjection(
            clip_style='single_token',
            token_dim=self.latent_dim,
            clip_token_index=1,
            num_token=3,
        )
        self.clip_loss_type = 'symmetric'
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.ema_callback_idx = None

        # Load pretrained checkpoint
        if ckpt_path is not None:
            self.load_pretrained_parameters(ckpt_path)
        else:
            pass
            # raise ValueError('[MDTDomainAdaptVisualEncoder] ckpt_path must be provided!')

        # Create model copies for domain adaptation AFTER loading pretrained weights
        self.source_static_resnet = copy.deepcopy(self.static_resnet)
        self.source_gripper_resnet = copy.deepcopy(self.gripper_resnet)
        self.source_model = copy.deepcopy(self.model)
        self.placeholder_param = torch.nn.Parameter(torch.ones([1]))
        # For domain adaptation
        self.use_da_act: bool = domain_adapt.use_da_act
        self.da_loss = hydra.utils.instantiate(domain_adapt.visual_da).to(self.device)
        self.da_2_loss = hydra.utils.instantiate(domain_adapt.visual_da).to(self.device)
        if self.use_da_act:
            self.da_act_loss = hydra.utils.instantiate(domain_adapt.action_da).to(self.device)
        self.cache_da_d_loss = 0.
        self.cache_wdist = 0.
        self.cache_da_g_loss = 0.
        # For visualization
        self.cache_s_enc = []
        self.cache_t_enc = []
        self.cache_s_emb = []
        self.cache_t_emb = []
        self.cache_s_output = []
        self.cache_t_output = []
        self.cache_s_action_gt = []
        self.cache_t_action_gt = []

    def load_pretrained_parameters(self, ckpt_path):
        """
        Load the pretrained parameters from the provided path.
        """
        print("Loading pretrained parameters")
        checkpoint_data = torch.load(ckpt_path)
        '''if 'callbacks'''
        if "ema_weights" in checkpoint_data['callbacks']['EMA']:
            ema_weights_list = checkpoint_data['callbacks']['EMA']['ema_weights']

            # Model's params dict
            model_weights_dict = {name: v for name, v in self.named_parameters()}
            popped_tiny_names = ['cross_att.bias', 'attn.bias']
            model_need_loading_weights_dict = {name: None for name, v in self.named_parameters()}
            missing_keys, unexpected_keys = [], []
            for tiny_name in popped_tiny_names:
                for name in model_need_loading_weights_dict.keys():
                    if tiny_name in name:
                        model_need_loading_weights_dict.pop(name)
                        missing_keys.append(name)

            # Convert list of tensors to a state_dict format
            ema_weights_dict = {name: ema_weights_list[i] for i, (name, _) in enumerate(model_need_loading_weights_dict.items())}
            # missing_keys, unexpected_keys = self.load_state_dict(ema_weights_dict)
            for name, v in ema_weights_dict.items():
                if name in model_weights_dict.keys():
                    model_weights_dict[name] = v
                else:
                    unexpected_keys.append(name)
            self.load_state_dict(model_weights_dict)
            print(f"Successfully loaded EMA weights from checkpoint! "
                  f"missing: {missing_keys}, unexpected: {unexpected_keys}")
        else:
            self.load_state_dict(checkpoint_data['state_dict'])
        print("Successfully loaded weights from checkpoint!")

    def configure_optimizers(self):
        """
        Initialize optimizers and learning rate schedulers based on model configuration.
        """
        # Configuration for models using transformer weight decay
        g_optim_groups = []
        g_act_optim_groups = []
        d_optim_groups = []

        self.set_requires_grad(self.visual_goal, False)
        self.set_requires_grad(self.source_static_resnet, False)
        self.set_requires_grad(self.source_gripper_resnet, False)
        self.set_requires_grad(self.source_model, False)

        self.set_requires_grad(self.gen_img, False)
        # self.set_requires_grad(self.clip_proj, False)
        # self.logit_scale.requires_grad = False

        self.set_requires_grad(self.model, False)
        if self.use_da_act:
            self.set_requires_grad(self.model, True)
            self.model.inner_model.freeze_backbone()
            g_act_optim_groups.extend([
                {"params": self.model.inner_model.trainable_params(), "weight_decay": self.optimizer_config.transformer_weight_decay},
                # {"params": self.gen_img.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
            ])
        else:
            g_act_optim_groups.extend([
                {"params": self.placeholder_param,"weight_decay": self.optimizer_config.transformer_weight_decay}
            ])
        g_optim_groups.extend([
            # {"params": self.clip_proj.parameters(), "weight_decay": self.optimizer_config.obs_encoder_weight_decay},
            # {"params": self.logit_scale, "weight_decay": self.optimizer_config.obs_encoder_weight_decay},
        ])

        self.set_requires_grad(self.da_loss, True)
        self.set_requires_grad(self.da_2_loss, True)
        d_optim_groups.extend([
            {"params": self.da_loss.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
            {"params": self.da_2_loss.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
        ])
        if self.use_da_act:
            self.set_requires_grad(self.da_act_loss, True)
            d_optim_groups.extend([
                {"params": self.da_act_loss.parameters(),
                 "weight_decay": self.optimizer_config.transformer_weight_decay},
            ])

        self.set_requires_grad(self.static_resnet, True)
        self.set_requires_grad(self.gripper_resnet, True)
        self.static_resnet.freeze_backbone()
        self.gripper_resnet.freeze_backbone()
        g_optim_groups.extend([
            {"params": self.static_resnet.trainable_params(), "weight_decay": self.optimizer_config.transformer_weight_decay},
            {"params": self.gripper_resnet.trainable_params(), "weight_decay": self.optimizer_config.transformer_weight_decay},
        ])

        # g_optimizer = torch.optim.AdamW(g_optim_groups, lr=self.optimizer_config.learning_rate,
        #                                 betas=self.optimizer_config.betas)
        # d_optimizer = torch.optim.AdamW(d_optim_groups, lr=self.optimizer_config.learning_rate,
        #                                 betas=self.optimizer_config.betas)
        g_optimizer = torch.optim.AdamW(g_optim_groups, lr=self.optimizer_config.learning_rate,
                                        betas=self.optimizer_config.betas)
        g_act_optimizer = torch.optim.AdamW(g_act_optim_groups, lr=self.optimizer_config.learning_rate,
                                            betas=self.optimizer_config.betas)
        d_optimizer = torch.optim.AdamW(d_optim_groups, lr=self.optimizer_config.da_lr,
                                        betas=self.optimizer_config.da_betas)

        # Optionally initialize the scheduler
        if self.use_lr_scheduler:
            da_lr_configs = OmegaConf.create(self.lr_scheduler.da_lr_scheduler)
            g_lr_configs = OmegaConf.create(self.lr_scheduler.enc_lr_scheduler)
            g_act_lr_configs = OmegaConf.create(self.lr_scheduler.enc_lr_scheduler)
            g_scheduler = TriStageLRScheduler(g_optimizer, g_lr_configs)
            g_lr_scheduler = {
                "scheduler": g_scheduler,
                "interval": 'step',
                "frequency": 1,
            }
            g_act_scheduler = TriStageLRScheduler(g_act_optimizer, g_act_lr_configs)
            g_act_lr_scheduler = {
                "scheduler": g_act_scheduler,
                "interval": 'step',
                "frequency": 1,
            }
            d_scheduler = TriStageLRScheduler(d_optimizer, da_lr_configs)
            d_lr_scheduler = {
                "scheduler": d_scheduler,
                "interval": 'step',
                "frequency": 1,
            }
            return (
                {"optimizer": g_optimizer,
                 "lr_scheduler": g_lr_scheduler,
                 },
                {"optimizer": g_act_optimizer,
                 "lr_scheduler": g_act_lr_scheduler,
                },
                {"optimizer": d_optimizer,
                 "lr_scheduler": d_lr_scheduler,
                 },
            )
        else:
            return g_optimizer, g_act_optimizer, d_optimizer

    @staticmethod
    def set_requires_grad(model, requires_grad=True):
        for param in model.parameters():
            param.requires_grad = requires_grad
        if requires_grad:
            model.train()
        else:
            model.eval()

    def on_before_zero_grad(self, optimizer=None):
        total_grad_norm = 0.0
        total_param_norm = 0.0
        all_none = True
        for name, p in self.named_parameters():  # ori:self.model.parameters()
            if 'da_loss' in name:
                continue
            elif 'da_2_loss' in name:
                continue
            elif 'da_act_loss' in name:
                continue
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item() ** 2
                all_none = False
            total_param_norm += p.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        total_param_norm = total_param_norm ** 0.5

        if all_none:
            print('[Warning] All grads are None!!!')

        self.log("train/grad_norm", total_grad_norm, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train/param_norm", total_param_norm, on_step=True, on_epoch=False, sync_dist=True)

    def clip_extra_forward(self, perceptual_emb, latent_goal, actions, sigmas, noise):

        self.model.train()
        noised_input = actions + noise * append_dims(sigmas, actions.ndim)
        context = self.model.forward_context_only(perceptual_emb, noised_input, latent_goal, sigmas)
        return context

    def training_step(self, batch: Dict[str, Dict], batch_idx: int,
                      dataloader_idx: int = 0) -> torch.Tensor:  # type: ignore
        """
        Compute and return the training loss for the MDT Agent.
        The training loss consists of the score matching loss of the diffusion model
        and the contrastive loss of the CLIP model for the multimodal encoder.

        Args:
            batch: Dictionary containing the batch data for each modality.
            batch_idx: Index of the batch. used for compatibility with pytorch lightning.
            dataloader_idx: Index of the dataloader. used for compatibility with pytorch lightning.

        Returns:
            loss tensor
        """
        g_opt, g_act_opt, d_opt = self.optimizers(use_pl_optimizer=True)  # pl_optimizer doesn't support AMP training
        g_sch, g_act_sch, d_sch = self.lr_schedulers()

        total_loss, action_loss, cont_loss, id_loss, img_gen_loss, da_d_loss, da_g_loss = (
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
        )
        losses = {
            'total_loss': total_loss,
            'action_loss': action_loss,
            'cont_loss': cont_loss,
            'img_gen_loss': img_gen_loss,
            'da_d_loss': da_d_loss,
            'da_g_loss': da_g_loss,
            'da_g_act_loss': torch.tensor(0.0).to(self.device),
        }
        encoders_dict = {}
        batch_size: Dict[str, int] = {}
        s_batch_len = 0
        t_batch_len = 0
        total_bs = 0
        # s_perceptual_emb = None
        # t_perceptual_emb = None
        s_latent_encoder_emb_dict = {}
        t_latent_encoder_emb_dict = {}
        s_latent_gripper_emb_dict = {}
        t_latent_gripper_emb_dict = {}
        s_latent_action_emb_dict = {}
        t_latent_action_emb_dict = {}
        s_feat_dict = {}
        t_feat_dict = {}
        s_action_gt_dict = {}
        t_action_gt_dict = {}
        rand_noise = None
        for self.modality_scope, dataset_batch in batch.items():
            # if 'lang' in self.modality_scope:  # skip:'lang_source', 'lang_target'
            #     continue
            # if 'source' in self.modality_scope:
            #     continue
            if dataset_batch is not None:
                rand_noise = torch.randn_like(dataset_batch["actions"]) if rand_noise is None else rand_noise

            if 'source' in self.modality_scope:
                # Compute the required embeddings
                s_perceptual_emb, latent_goal, image_latent_goal = self.compute_input_embeddings(
                    dataset_batch, is_target=False)
                # 'static' or 'gripper':(bs,1,512)
                s_batch_len += 1

                # Compute diffusion loss without actions, just for sigmas
                _, sigmas, noise = self.diffusion_loss(
                    s_perceptual_emb,
                    latent_goal,  # (64,512)
                    rand_noise,  # no need to calculate loss
                    is_target=False,
                )  # will call enc_only_forward() and dec_only_forward()
                latent_encoder_emb = self.source_model.inner_model.latent_encoder_emb
                latent_action_emb = self.source_model.inner_model.cache_action_emb
                action_output = self.source_model.inner_model.cache_action_output
                # self.source_model.inner_model.enc_only_forward(
                #     s_perceptual_emb,
                #     actions=None,
                #     goals=latent_goal,
                #     sigma=sigmas,
                # )  # encoder doesn't use actions
                # latent_encoder_emb = self.source_model.inner_model.latent_encoder_emb

                # s_latent_encoder_emb_dict[self.modality_scope[:-len('_source')]] = latent_encoder_emb
                # s_latent_encoder_emb_dict[self.modality_scope[:-len('_source')]] = torch.cat([
                #     s_perceptual_emb['static'], s_perceptual_emb['gripper']
                # ], dim=-1)  # (bs,1,1024)
                s_latent_encoder_emb_dict[self.modality_scope[:-len('_source')]] = s_perceptual_emb[
                    'static']  # (bs,1,512)
                s_latent_gripper_emb_dict[self.modality_scope[:-len('_source')]] = s_perceptual_emb['gripper']
                s_latent_action_emb_dict[self.modality_scope[:-len('_source')]] = latent_action_emb
                s_feat_dict[self.modality_scope[:-len('_source')]] = action_output
                s_action_gt_dict[self.modality_scope[:-len('_source')]] = dataset_batch["actions"]

            elif 'target' in self.modality_scope:
                t_perceptual_emb, latent_goal, image_latent_goal = self.compute_input_embeddings(
                    dataset_batch, is_target=True)
                t_batch_len += 1

                # Compute diffusion loss without actions, just for sigmas
                _, sigmas, noise = self.diffusion_loss(
                    t_perceptual_emb,
                    latent_goal,
                    rand_noise,  # no need to calculate loss
                    is_target=True,
                )
                latent_encoder_emb = self.model.inner_model.latent_encoder_emb
                latent_action_emb = self.model.inner_model.cache_action_emb
                action_output = self.model.inner_model.cache_action_output
                # self.model.inner_model.enc_only_forward(
                #     t_perceptual_emb,
                #     actions=None,
                #     goals=latent_goal,
                #     sigma=sigmas,
                # )  # encoder doesn't use actions
                # latent_encoder_emb = self.model.inner_model.latent_encoder_emb

                # # Compute diffusion loss for DEBUG (DO NOT use in method!)
                # diff_loss, sigmas, noise = self.diffusion_loss(
                #     t_perceptual_emb,
                #     latent_goal,
                #     dataset_batch['actions'],
                #     is_target=True,
                #     is_da=False,
                # )
                # action_loss += diff_loss

                # # Compute the masked generative foresight loss (only for target)
                # if not isinstance(self.gen_img, NoEncoder):
                #     rgb_static_goal = dataset_batch["rgb_obs"]['gen_static']
                #     rgb_gripper_goal = dataset_batch["rgb_obs"]['gen_gripper']
                #     img_gen_frame_diff = dataset_batch[
                #         'future_frame_diff'] if "future_frame_diff" in dataset_batch else 3
                #     # combine both goal images
                #     rgb_pred_goal = torch.cat([rgb_static_goal, rgb_gripper_goal], dim=1)
                #     img_gen_embed = latent_encoder_emb
                #     img_gen_loss_part = self.compute_img_gen_loss(img_gen_embed, rgb_pred_goal,
                #                                                   img_gen_frame_diff=img_gen_frame_diff)
                #     img_gen_loss += img_gen_loss_part * self.masked_beta

                # # Compute the Contrastive Latent Alignment Loss (only for target)
                # cont_loss_part = self.compute_contrastive_loss(
                #     t_perceptual_emb,
                #     latent_goal,
                #     image_latent_goal,
                #     dataset_batch,
                #     sigmas,
                #     noise
                # )
                # cont_loss += self.cont_alpha * cont_loss_part

                # t_latent_encoder_emb_dict[self.modality_scope[:-len('_target')]] = latent_encoder_emb
                # t_latent_encoder_emb_dict[self.modality_scope[:-len('_target')]] = torch.cat([
                #     t_perceptual_emb['static'], t_perceptual_emb['gripper']
                # ], dim=-1)  # (bs,1,1024)
                t_latent_encoder_emb_dict[self.modality_scope[:-len('_target')]] = t_perceptual_emb[
                    'static']  # (bs,1,512)
                t_latent_gripper_emb_dict[self.modality_scope[:-len('_target')]] = t_perceptual_emb['gripper']
                t_latent_action_emb_dict[self.modality_scope[:-len('_target')]] = latent_action_emb  # (bs,10,512)
                t_feat_dict[self.modality_scope[:-len('_target')]] = action_output
                t_action_gt_dict[self.modality_scope[:-len('_target')]] = dataset_batch["actions"]

            else:
                raise KeyError(f'[MDTDomainAdaptVisualEncoder] batch key:{self.modality_scope} not supported')

            batch_size[self.modality_scope] = dataset_batch["actions"].shape[0]
            total_bs += dataset_batch["actions"].shape[0]

        # divide accumulated gradients by number of datasets
        batch_len = s_batch_len + t_batch_len

        # sort dict
        def sort_dict(dict1):
            return {key: dict1[key] for key in sorted(dict1.keys())}

        s_latent_encoder_emb_dict = sort_dict(s_latent_encoder_emb_dict)
        t_latent_encoder_emb_dict = sort_dict(t_latent_encoder_emb_dict)
        s_latent_action_emb_dict = sort_dict(s_latent_action_emb_dict)
        t_latent_action_emb_dict = sort_dict(t_latent_action_emb_dict)

        t_feat_for_da = torch.cat([v for v in t_latent_encoder_emb_dict.values()], dim=0)
        s_feat_for_da = torch.cat([v for v in s_latent_encoder_emb_dict.values()], dim=0)
        t_feat_for_da_2 = torch.cat([v for v in t_latent_gripper_emb_dict.values()], dim=0)
        s_feat_for_da_2 = torch.cat([v for v in s_latent_gripper_emb_dict.values()], dim=0)
        t_feat_for_da_act = torch.cat([v for v in t_latent_action_emb_dict.values()], dim=0)
        s_feat_for_da_act = torch.cat([v for v in s_latent_action_emb_dict.values()], dim=0)

        ''' 1. Update discriminator '''
        if len(self.cache_t_emb) < 20 and len(self.cache_s_emb) < 20:
            t_keys = list(t_latent_action_emb_dict.keys())
            s_keys = list(s_latent_action_emb_dict.keys())
            t_key = t_keys[-1]
            bs = t_latent_action_emb_dict[t_key].shape[0]
            last_dim = t_latent_action_emb_dict[t_key].shape[-1]
            # print(t_keys, s_keys, t_latent_action_emb_dict[t_key].shape)
            self.cache_t_enc.append(t_latent_encoder_emb_dict[t_key].detach().float().cpu().reshape(bs, -1).numpy())
            self.cache_s_enc.append(s_latent_encoder_emb_dict[t_key].detach().float().cpu().reshape(bs, -1).numpy())
            self.cache_t_emb.append(t_latent_action_emb_dict[t_key].detach().cpu().reshape(bs, -1).numpy())
            self.cache_s_emb.append(s_latent_action_emb_dict[t_key].detach().cpu().reshape(bs, -1).numpy())
            # self.cache_t_emb.append(t_latent_action_emb_dict[t_key].detach().cpu().reshape(-1, last_dim).numpy())
            # self.cache_s_emb.append(s_latent_action_emb_dict[t_key].detach().cpu().reshape(-1, last_dim).numpy())
            self.cache_t_output.append(t_feat_dict[t_key].detach().float().cpu().reshape(bs, -1).numpy())
            self.cache_s_output.append(s_feat_dict[t_key].detach().float().cpu().reshape(bs, -1).numpy())
            self.cache_t_action_gt.append(t_action_gt_dict[t_key].detach().float().cpu().reshape(bs, -1).numpy())
            self.cache_s_action_gt.append(s_action_gt_dict[t_key].detach().float().cpu().reshape(bs, -1).numpy())

        from mdt.datasets.utils.debug_utils import TSNEHelper
        if (os.environ.get("LOCAL_RANK", "0") == "0" and batch_idx % 400 == -1 and
                len(self.cache_t_emb) >= 20 and len(self.cache_s_emb) >= 20):
            epoch_idx = self.current_epoch
            tsne_inputs = np.concatenate(self.cache_t_enc + self.cache_s_enc, axis=0)
            helper = TSNEHelper(tsne_inputs)
            helper.plot_tsne(f'visual_enc_{epoch_idx:02d}_{batch_idx:05d}')

            # tsne_inputs = np.concatenate(self.cache_t_emb + self.cache_s_emb, axis=0)
            # helper = TSNEHelper(tsne_inputs)
            # helper.plot_tsne(f'action_embedding_{epoch_idx:02d}_{batch_idx:05d}')

            tsne_inputs = np.concatenate(self.cache_t_output + self.cache_s_output, axis=0)
            helper = TSNEHelper(tsne_inputs)
            helper.plot_tsne(f'action_output_{epoch_idx:02d}_{batch_idx:05d}')

            # tsne_inputs = np.concatenate(self.cache_t_action_gt + self.cache_s_action_gt, axis=0)
            # helper = TSNEHelper(tsne_inputs)
            # helper.plot_tsne(f'action_gt_{epoch_idx:02d}_{batch_idx:05d}')

            self.cache_t_enc = []
            self.cache_s_enc = []
            self.cache_t_action_gt = []
            self.cache_s_action_gt = []
            self.cache_t_output = []
            self.cache_s_output = []
            self.cache_t_emb = []
            self.cache_s_emb = []

        da_loss_dict = self.da_loss.forward(
            t_feat_for_da.clone().detach(),  # avoid grad of G_target
            s_feat_for_da.detach(),  # avoid grad of G_source
            is_discriminator_batch=True,
        )
        da_d_loss = da_loss_dict['loss']
        w_dist = da_loss_dict['w_dist']
        gp = da_loss_dict['gp']  # just for log
        da_d_loss = da_d_loss / 1  # choose 1 from 2
        losses['da_d_loss'] += da_d_loss
        losses['w_dist_1'] = w_dist
        losses['gp_1'] = gp

        da_2_loss_dict = self.da_2_loss.forward(
            t_feat_for_da_2.clone().detach(),  # avoid grad of G_target
            s_feat_for_da_2.detach(),  # avoid grad of G_source
            is_discriminator_batch=True,
        )
        da_d_2_loss = da_2_loss_dict['loss']
        w_dist = da_2_loss_dict['w_dist']
        gp = da_2_loss_dict['gp']  # just for log
        losses['da_d_loss'] += da_d_2_loss / 1
        losses['w_dist_2'] = w_dist
        losses['gp_2'] = gp

        if self.use_da_act:
            da_act_loss_dict = self.da_act_loss.forward(
                t_feat_for_da_act.clone().detach(),  # avoid grad of G_target
                s_feat_for_da_act.detach(),  # avoid grad of G_source
                is_discriminator_batch=True,
            )
            da_d_act_loss = da_act_loss_dict['loss']
            w_dist = da_2_loss_dict['w_dist']
            gp = da_2_loss_dict['gp']  # just for log
            losses['da_d_loss'] += da_d_act_loss / 1
            losses['w_dist_act'] = w_dist
            losses['gp_act'] = gp

        d_opt.zero_grad()
        self.manual_backward(losses['da_d_loss'])
        d_opt.step()
        d_sch.step()

        ''' 2. Update generator '''
        da_loss_dict = self.da_loss.forward(
            t_feat_for_da,  # update G_target
            s_feat_for_da.detach(),  # avoid grad of G_source
            is_discriminator_batch=False,
        )
        da_g_loss = da_loss_dict['loss']
        gp = da_loss_dict['gp']  # just for log
        da_g_loss = da_g_loss / 1  # choose 1 from 2
        losses['da_g_loss'] += da_g_loss

        da_2_loss_dict = self.da_2_loss.forward(
            t_feat_for_da_2,  # update G_target
            s_feat_for_da_2.detach(),  # avoid grad of G_source
            is_discriminator_batch=False,
        )
        da_g_2_loss = da_2_loss_dict['loss']
        gp = da_2_loss_dict['gp']  # just for log
        losses['da_g_loss'] += da_g_2_loss / 1

        if self.use_da_act:
            da_act_loss_dict = self.da_act_loss.forward(
                t_feat_for_da_act,  # update G_target
                s_feat_for_da_act.detach(),  # avoid grad of G_source
                is_discriminator_batch=False,
            )
            da_g_act_loss = da_act_loss_dict['loss']
            gp = da_act_loss_dict['gp']  # just for log
            da_g_act_loss += da_g_act_loss / 1
            losses['da_g_act_loss'] += da_g_act_loss / 1

            g_act_opt.zero_grad()
            self.manual_backward(losses['da_g_act_loss'], retain_graph=True)
            g_act_opt.step()
            g_act_sch.step()

        losses['cont_loss'] += cont_loss / t_batch_len  # used
        losses['action_loss'] += action_loss / batch_len  # NOT used
        losses['img_gen_loss'] += img_gen_loss / t_batch_len  # used

        backward_loss = losses['da_g_loss'] + losses['img_gen_loss'] + losses['cont_loss'] + losses['action_loss']
        losses['total_loss'] = backward_loss + losses['da_g_act_loss']

        g_opt.zero_grad()
        self.manual_backward(backward_loss)
        if not self.automatic_optimization:
            self.on_before_zero_grad()
        g_opt.step()
        g_sch.step()

        # Log the metrics
        self._log_training_metrics(losses, total_bs)

        return total_loss

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Dict], batch_idx: int, dataloader_idx: int = 0) -> Dict[
        str, torch.Tensor]:  # type: ignore
        """
        Compute and log the validation losses and additional metrics.
        During the validation step, the diffusion model predicts the next action sequence given the current state

        Args:
            batch: Dictionary containing the batch data for each modality.
            batch_idx: Index of the batch. used for compatibility with pytorch lightning.
            dataloader_idx: Index of the dataloader. used for compatibility with pytorch lightning.

        Returns:
            Dictionary containing the sampled plans of plan recognition and plan proposal networks, as well as the
            episode indices.
        """
        output = {}
        val_total_act_loss_pp = torch.tensor(0.0).to(self.device)
        for self.modality_scope, dataset_batch in batch.items():
            if "source" in self.modality_scope:
                continue
            # Compute the required embeddings
            perceptual_emb, latent_goal, image_latent_goal = self.compute_input_embeddings(dataset_batch)

            # predict the next action sequence
            action_pred = self.denoise_actions(
                torch.zeros_like(latent_goal).to(latent_goal.device),
                perceptual_emb,
                latent_goal,
                inference=True,
            )
            # compute the mse action loss
            pred_loss = torch.nn.functional.mse_loss(action_pred, dataset_batch["actions"])
            latent_encoder_emb = self.model.inner_model.latent_encoder_emb
            val_total_act_loss_pp += pred_loss

            # next compute the image generation loss
            if not isinstance(self.gen_img, NoEncoder):
                rgb_static_goal = dataset_batch["rgb_obs"]['gen_static']
                rgb_gripper_goal = dataset_batch["rgb_obs"]['gen_gripper']
                img_gen_frame_diff = dataset_batch['future_frame_diff'] if "future_frame_diff" in dataset_batch else 3
                # combine both goal images
                rgb_pred_goal = torch.cat([rgb_static_goal, rgb_gripper_goal], dim=1)

                img_gen_embed = latent_encoder_emb

                img_gen_loss = self.compute_img_gen_loss(
                    img_gen_embed,
                    rgb_pred_goal,
                    store_img=False,
                    batch_idx=batch_idx,
                    img_gen_frame_diff=img_gen_frame_diff,
                )
            else:
                img_gen_loss = torch.tensor(0.0).to(self.device)

            self._log_validation_metrics(pred_loss, img_gen_loss, val_total_act_loss_pp)

            output[f"idx_{self.modality_scope}"] = dataset_batch["idx"]
            output["validation_loss"] = val_total_act_loss_pp
        return output

    def compute_input_embeddings(self, dataset_batch, is_target: bool = True):
        """
        Compute the required embeddings for the visual ones and the latent goal.
        """
        # 1. extract the revelant visual observations
        latent_goal = None
        rgb_static_goal = dataset_batch["rgb_obs"]['rgb_static'][:, -1]
        rgb_static = dataset_batch["rgb_obs"]['rgb_static'][:, :-1]

        rgb_gripper = dataset_batch["rgb_obs"]['rgb_gripper'][:, :-1]
        modality = "vis"

        # 2. Compute the latent goal embedding for the visual goal
        if not isinstance(self.visual_goal, NoEncoder):
            latent_goal = self.visual_goal(rgb_static_goal).to(rgb_static.dtype)

        lang_text = dataset_batch["lang_text"] if "lang" in self.modality_scope else None

        # 3. we compute the language goal if the language modality is in the scope
        if "lang" in self.modality_scope:
            modality = "lang"
            image_latent_goal = latent_goal.to(rgb_static.dtype)
            if self.use_text_not_embedding:
                latent_goal = self.language_goal(dataset_batch["lang_text"]).to(rgb_static.dtype)
            else:
                latent_goal = self.language_goal(dataset_batch["lang"]).to(rgb_static.dtype)
        else:
            image_latent_goal = None

        perceptual_emb = self.embed_visual_obs(rgb_static, rgb_gripper, is_target)
        perceptual_emb['modality'] = modality
        return perceptual_emb, latent_goal, image_latent_goal

    def embed_visual_obs(self, rgb_static, rgb_gripper, is_target=True):
        # reshape rgb_static and rgb_gripper
        rgb_static = einops.rearrange(rgb_static, 'b t c h w -> (b t) c h w')
        rgb_gripper = einops.rearrange(rgb_gripper, 'b t c h w -> (b t) c h w')
        if is_target:
            static_resnet = self.static_resnet
            gripper_resnet = self.gripper_resnet
        else:
            static_resnet = self.source_static_resnet
            gripper_resnet = self.source_gripper_resnet

        static_tokens = static_resnet(rgb_static)
        gripper_tokens = gripper_resnet(rgb_gripper)
        static_tokens = einops.rearrange(static_tokens, 'b (t d) -> b t d', t=1)
        gripper_tokens = einops.rearrange(gripper_tokens, 'b (t d) -> b t d', t=1)
        token_seq = {
            'static': static_tokens,
            'gripper': gripper_tokens,
        }
        return token_seq

    def clip_extra_forward(self, perceptual_emb, latent_goal, actions, sigmas, noise):
        self.model.train()
        noised_input = actions + noise * append_dims(sigmas, actions.ndim)
        context = self.model.forward_context_only(perceptual_emb, noised_input, latent_goal, sigmas)
        return context

    def compute_img_gen_loss(self, latent_embeddings, goal_img, store_img=False, img_gen_frame_diff=3, batch_idx=0):
        """
        Compute the image generation loss based on the provided embeddings and dataset batch.
        """
        if len(goal_img.shape) == 5:
            goal_img = goal_img.squeeze(1)
            # the goal is not to reconstruct all the details but to get the general shape
        # 1. predict the future image patches
        img_gen_pred, mask, restore_idxs, visible_patches = self.gen_img(latent_embeddings, goal_img,
                                                                         img_gen_frame_diff)
        # 2. compute the loss
        img_gen_loss = self.gen_img.compute_loss(goal_img, img_gen_pred, mask, restore_idxs)
        if store_img:
            file_path = os.getcwd() + f'/img_gen_pred_{batch_idx}.png'
            self.gen_img.reconstruct_image(
                predictions=img_gen_pred,
                goal_images=goal_img,
                mask=mask,
                restore_idxs=restore_idxs,
                file_path=file_path,
            )
            try:
                self.logger.experiment.log({f"generated_img_{batch_idx}": wandb.Image(os.path.abspath(file_path))})
            except Exception as e:
                print(f"An error occurred while saving or logging image: {e}")
                # Optionally, you can log the error to wandb as well
                self.logger.experiment.log({"error": str(e)})

        return img_gen_loss

    def compute_contrastive_loss(self, perceptual_emb, latent_goal, image_latent_goal, dataset_batch, sigma, noise):
        """
        Compute the contrastive loss based on the provided embeddings and dataset batch.
        """
        if "lang" in self.modality_scope:
            latent_language_embed = self.model.inner_model.latent_encoder_emb

            latent_vis_embed = self.clip_extra_forward(
                perceptual_emb,
                image_latent_goal,
                dataset_batch["actions"],
                sigma,  # Assuming you don't need sigmas and noise here
                noise
            )
            latent_language_embed = self.clip_proj(latent_language_embed)
            latent_vis_embed = self.clip_proj(latent_vis_embed)

            is_distributed = self.trainer.global_rank >= 0 and dist.is_initialized()

            if is_distributed and self.use_distributed_clip:

                all_latent_vis_embed = self.all_gather(latent_vis_embed, sync_grads=True)
                all_latent_language_embed = self.all_gather(latent_language_embed, sync_grads=True)
                all_latent_language_embed = einops.rearrange(all_latent_language_embed, 'n b d -> (n b) d')
                all_latent_vis_embed = einops.rearrange(all_latent_vis_embed, 'n b d -> (n b) d')

            else:
                all_latent_vis_embed = latent_vis_embed
                all_latent_language_embed = latent_language_embed

            lang_text = dataset_batch["lang_text"] if "lang_text" in dataset_batch else None

            # Compute contrastive loss with gathered embeddings
            cont_loss_part = self.cont_loss(
                all_latent_vis_embed,
                all_latent_language_embed,
                mode=self.clip_loss_type,
                lang_text=lang_text
            )

            return cont_loss_part
        else:
            return torch.tensor(0.0).to(self.device)  # Return a zero tensor if "lang" is not in the modality scope

    def _log_training_metrics(self, log_dict, total_bs):
        """
        Log the training metrics.
        """
        for k, v in log_dict.items():
            self.log(f"train/{k}", v.clone().detach(),
                     on_step=True, on_epoch=False, sync_dist=True, batch_size=total_bs)

    def _log_validation_metrics(self, pred_loss, img_gen_loss, val_total_act_loss_pp):
        """
        Log the validation metrics.
        """
        self.log(f"val_act/{self.modality_scope}_act_loss_pp", pred_loss, sync_dist=True)
        self.log(
            "val_act/action_loss",
            val_total_act_loss_pp / len(self.trainer.datamodule.modalities),  # type:ignore
            sync_dist=True,
        )
        self.log(f"val_act/img_gen_loss_pp", img_gen_loss, sync_dist=True)

    def diffusion_loss(
            self,
            perceptual_emb: dict,
            latent_goal: torch.Tensor,
            actions: torch.Tensor,  # gt
            is_target: bool = True,
            is_da: bool = True,
    ) -> torch.Tensor:
        """
        Computes the score matching loss given the perceptual embedding, latent goal, and desired actions.
        """
        bs = perceptual_emb['static'].shape[0]
        if bs != actions.shape[0]:
            actions = actions[:bs]
        if not is_target:  # source
            self.source_model.eval()
            sigmas = self.make_sample_density()(shape=(len(actions),), device=self.device).to(self.device)
            noise = torch.randn_like(actions).to(self.device)
            loss, _ = self.source_model.loss(perceptual_emb, actions, latent_goal, noise, sigmas, is_da=is_da)
            # loss = 0.
        else:
            self.model.train()
            sigmas = self.make_sample_density()(shape=(len(actions),), device=self.device).to(self.device)
            noise = torch.randn_like(actions).to(self.device)
            loss, _ = self.model.loss(perceptual_emb, actions, latent_goal, noise, sigmas, is_da=is_da)
            # loss = 0.
        return loss, sigmas, noise

    def denoise_actions(  # type: ignore
            self,
            latent_plan: torch.Tensor,
            perceptual_emb: torch.Tensor,
            latent_goal: torch.Tensor,
            inference: Optional[bool] = False,
            extra_args={}
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Denoise the next sequence of actions
        """
        if inference:
            sampling_steps = self.num_sampling_steps
        else:
            sampling_steps = 10
        self.model.eval()
        if len(latent_goal.shape) < len(
                perceptual_emb['static'].shape if isinstance(perceptual_emb, dict) else perceptual_emb.shape):
            latent_goal = latent_goal.unsqueeze(1)  # .expand(-1, seq_len, -1)
        input_state = perceptual_emb
        sigmas = self.get_noise_schedule(sampling_steps, self.noise_scheduler)
        if len(latent_goal.shape) == 2:
            goal = einops.rearrange(goal, 'b d -> 1 b d')

        x = torch.randn((len(latent_goal), self.act_window_size, 7), device=self.device) * self.sigma_max

        actions = self.sample_loop(sigmas, x, input_state, latent_goal, latent_plan, self.sampler_type, extra_args)

        return actions

    def make_sample_density(self):
        """
        Generate a sample density function based on the desired type for training the model
        We mostly use log-logistic as it has no additional hyperparameters to tune.
        """
        sd_config = []
        if self.sigma_sample_density_type == 'lognormal':
            loc = self.sigma_sample_density_mean  # if 'mean' in sd_config else sd_config['loc']
            scale = self.sigma_sample_density_std  # if 'std' in sd_config else sd_config['scale']
            return partial(utils.rand_log_normal, loc=loc, scale=scale)

        if self.sigma_sample_density_type == 'loglogistic':
            loc = sd_config['loc'] if 'loc' in sd_config else math.log(self.sigma_data)
            scale = sd_config['scale'] if 'scale' in sd_config else 0.5
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)

        if self.sigma_sample_density_type == 'loguniform':
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_log_uniform, min_value=min_value, max_value=max_value)

        if self.sigma_sample_density_type == 'uniform':
            return partial(utils.rand_uniform, min_value=self.sigma_min, max_value=self.sigma_max)

        if self.sigma_sample_density_type == 'v-diffusion':
            min_value = self.min_value if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_v_diffusion, sigma_data=self.sigma_data, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'discrete':
            sigmas = self.get_noise_schedule(self.num_sampling_steps * 1e5, 'exponential')
            return partial(utils.rand_discrete, values=sigmas)
        if self.sigma_sample_density_type == 'split-lognormal':
            loc = sd_config['mean'] if 'mean' in sd_config else sd_config['loc']
            scale_1 = sd_config['std_1'] if 'std_1' in sd_config else sd_config['scale_1']
            scale_2 = sd_config['std_2'] if 'std_2' in sd_config else sd_config['scale_2']
            return partial(utils.rand_split_log_normal, loc=loc, scale_1=scale_1, scale_2=scale_2)
        else:
            raise ValueError('Unknown sample density type')

    def sample_loop(
            self,
            sigmas,
            x_t: torch.Tensor,
            state: torch.Tensor,
            goal: torch.Tensor,
            latent_plan: torch.Tensor,
            sampler_type: str,
            extra_args={},
    ):
        """
        Main method to generate samples depending on the chosen sampler type. DDIM is the default as it works well in all settings.
        """
        s_churn = extra_args['s_churn'] if 's_churn' in extra_args else 0
        s_min = extra_args['s_min'] if 's_min' in extra_args else 0
        use_scaler = extra_args['use_scaler'] if 'use_scaler' in extra_args else False
        keys = ['s_churn', 'keep_last_actions']
        if bool(extra_args):
            reduced_args = {x: extra_args[x] for x in keys}
        else:
            reduced_args = {}
        if use_scaler:
            scaler = self.scaler
        else:
            scaler = None
        # ODE deterministic
        if sampler_type == 'lms':
            x_0 = sample_lms(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True, extra_args=reduced_args)
        # ODE deterministic can be made stochastic by S_churn != 0
        elif sampler_type == 'heun':
            x_0 = sample_heun(self.model, state, x_t, goal, sigmas, scaler=scaler, s_churn=s_churn, s_tmin=s_min,
                              disable=True)
        # ODE deterministic
        elif sampler_type == 'euler':
            x_0 = sample_euler(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # SDE stochastic
        elif sampler_type == 'ancestral':
            x_0 = sample_dpm_2_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
            # SDE stochastic: combines an ODE euler step with an stochastic noise correcting step
        elif sampler_type == 'euler_ancestral':
            x_0 = sample_euler_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm':
            x_0 = sample_dpm_2(self.model, state, x_t, goal, sigmas, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_adaptive':
            x_0 = sample_dpm_adaptive(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_fast':
            x_0 = sample_dpm_fast(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), len(sigmas),
                                  disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2s_ancestral':
            x_0 = sample_dpmpp_2s_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2m':
            x_0 = sample_dpmpp_2m(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2m_sde':
            x_0 = sample_dpmpp_sde(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'ddim':
            x_0 = sample_ddim(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2s':
            x_0 = sample_dpmpp_2s(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2_with_lms':
            x_0 = sample_dpmpp_2_with_lms(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        else:
            raise ValueError('desired sampler type not found!')
        return x_0

    def get_noise_schedule(self, n_sampling_steps, noise_schedule_type):
        """
        Get the noise schedule for the sampling steps. Describes the distribution over the noise levels from sigma_min to sigma_max.
        """
        if noise_schedule_type == 'karras':
            return get_sigmas_karras(n_sampling_steps, self.sigma_min, self.sigma_max, 7,
                                     self.device)  # rho=7 is the default from EDM karras
        elif noise_schedule_type == 'exponential':
            return get_sigmas_exponential(n_sampling_steps, self.sigma_min, self.sigma_max, self.device)
        elif noise_schedule_type == 'vp':
            return get_sigmas_vp(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 'linear':
            return get_sigmas_linear(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'cosine_beta':
            return cosine_beta_schedule(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 've':
            return get_sigmas_ve(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'iddpm':
            return get_iddpm_sigmas(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        raise ValueError('Unknown noise schedule type')

    def reset(self):
        """
        Call this at the beginning of a new rollout when doing inference.
        """
        self.plan = None
        self.latent_goal = None
        self.rollout_step_counter = 0

    def forward(self, obs, goal):
        """
        Method for doing inference with the model.
        """
        if 'lang' in goal:
            modality = 'lang'
            if self.use_text_not_embedding:
                # print(goal.keys())
                latent_goal = self.language_goal(goal["lang_text"])
                latent_goal = latent_goal.to(torch.float32)
            else:
                latent_goal = self.language_goal(goal["lang"]).unsqueeze(0).to(torch.float32).to(
                    obs["rgb_obs"]['rgb_static'].device)
        else:
            modality = 'vis'
            if self.use_delta_goal:
                perceptual_goal_emb = self.visual_goal(obs["rgb_obs"]['rgb_static'].squeeze(0))
            else:
                perceptual_goal_emb = self.visual_goal(obs["rgb_obs"]['rgb_static'][:, -1]).unsqueeze(1)  # [:, -1])

            latent_goal = perceptual_goal_emb

        rgb_static = obs["rgb_obs"]['rgb_static']
        rgb_gripper = obs["rgb_obs"]['rgb_gripper']

        perceptual_emb = self.embed_visual_obs(rgb_static, rgb_gripper)
        perceptual_emb['modality'] = modality

        act_seq = self.denoise_actions(
            torch.zeros_like(latent_goal).to(latent_goal.device),
            perceptual_emb,
            latent_goal,
            inference=True,
        )
        return act_seq

    def step(self, obs, goal):
        """
        Do one step of inference with the model. THis method handles the action chunking case.
        Our model is trained to predict a sequence of actions.
        We only compute the sequence once every self.multistep steps.

        Args:
            obs (dict): Observation from environment.
            goal (dict): Goal as visual observation or embedded language instruction.

        Returns:
            Predicted action.
        """
        if self.rollout_step_counter % self.multistep == 0:
            pred_action_seq = self(obs, goal)

            self.pred_action_seq = pred_action_seq

        current_action = self.pred_action_seq[0, self.rollout_step_counter]
        if len(current_action.shape) == 2:
            current_action = einops.rearrange(current_action, 'b d -> b 1 d')
        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.multistep:
            self.rollout_step_counter = 0

        return current_action

    def on_train_start(self) -> None:

        self.source_model.to(dtype=self.dtype)
        self.source_static_resnet.to(dtype=self.dtype)
        self.source_gripper_resnet.to(dtype=self.dtype)
        self.model.to(dtype=self.dtype)
        self.static_resnet.to(dtype=self.dtype)
        self.gripper_resnet.to(dtype=self.dtype)
        self.language_goal.to(dtype=self.dtype)
        self.visual_goal.to(dtype=self.dtype)
        self.gen_img.to(dtype=self.dtype)

        for idx, callback in enumerate(self.trainer.callbacks):
            if isinstance(callback, EMA):
                self.ema_callback_idx = idx
                break

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        logger.info(f"Finished training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")

    def clip_auxiliary_loss(self, image_features, lang_features, mode='symmetric', lang_text=None):
        # Normalize the features
        image_features = F.normalize(image_features, dim=-1)
        lang_features = F.normalize(lang_features, dim=-1)
        logit_scale = self.logit_scale.exp()

        # Compute the cosine similarity
        similarity_matrix = logit_scale * image_features @ lang_features.t()

        # InfoNCE loss
        labels = torch.arange(similarity_matrix.shape[0], device=image_features.device)
        infonce_loss = F.cross_entropy(similarity_matrix, labels)

        if mode == 'symmetric':
            similarity_matrix_lang_img = logit_scale * lang_features @ image_features.t()
            # similarity_matrix_lang_img.masked_fill_(~unique_mask, float('-inf'))
            infonce_loss_lang_img = F.cross_entropy(similarity_matrix_lang_img, labels)
            infonce_loss = (infonce_loss + infonce_loss_lang_img) / 2
        elif mode == 'img_to_text':
            pass  # already computed above
        elif mode == 'text_to_img':
            similarity_matrix = similarity_matrix.t()  # transpose for text-to-image
            infonce_loss = F.cross_entropy(similarity_matrix, labels)
        else:
            raise ValueError("Invalid mode. Expected one of: 'symmetric', 'img_to_text', 'text_to_img'.")
        return infonce_loss

    def on_validation_epoch_start(self) -> None:
        log_rank_0(f"Start validation epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        logger.info(f"Finished training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")

    def on_validation_epoch_start(self) -> None:
        log_rank_0(f"Start validation epoch {self.current_epoch}")


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)
