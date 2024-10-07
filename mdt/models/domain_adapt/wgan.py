import torch
import torch.nn as nn
from torch.autograd import Variable
import time as t
import os
from torchvision import utils
import torch.nn.functional as F


class Discriminator(torch.nn.Module):
    def __init__(self, channels: int = 3 * 384):
        super(Discriminator, self).__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        # self.main_module = nn.Sequential(
        #     # Image (Cx32x32)
        #     nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(num_features=256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     # State (256x16x16)
        #     nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(num_features=512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     # State (512x8x8)
        #     nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(num_features=1024),
        #     nn.LeakyReLU(0.2, inplace=True))
        #     # output of main module --> State (1024x4x4)
        self.main_module = nn.Sequential(
            nn.Linear(in_features=channels, out_features=384, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=384, out_features=384, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=384, out_features=256, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # self.output = nn.Sequential(
        #     # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
        #     nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))
        self.output = nn.Linear(in_features=256, out_features=1, bias=False)

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class WGAN_CP(torch.nn.Module):
    """
    Code refers to: https://github.dev/aadhithya/gan-zoo-pytorch/blob/master/models/wgan_gp.py
    """
    def __init__(self):
        super(WGAN_CP, self).__init__()
        self.discriminator = Discriminator()

    def forward(self, target_feat, source_feat=None, is_discriminator_batch: bool = True, gt_labels=None,):
        if source_feat is None:
            assert is_discriminator_batch, "source_feat should be given when is_discriminator_batch=True"
            source_feat = target_feat
        bs = source_feat.shape[0]
        source_feat = source_feat.view(bs, -1)
        target_feat = target_feat.view(bs, -1)
        device = source_feat.device

        if is_discriminator_batch:
            d_loss_real = self.discriminator(source_feat)
            d_loss_real = d_loss_real.mean(0).view(1)
            d_loss_fake = self.discriminator(target_feat)
            d_loss_fake = d_loss_fake.mean(0).view(1)
            loss = d_loss_real - d_loss_fake
        else:
            g_loss = self.discriminator(target_feat)
            g_loss = g_loss.mean().mean(0).view(1)
            loss = g_loss
        return loss


# StyleGAN2
def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1., bias = True):
        super(EqualLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class StyleVectorizer(nn.Module):
    def __init__(self, in_dim, out_dim, depth, lr_mul = 0.1):
        super(StyleVectorizer, self).__init__()

        layers = [
            torch.nn.LayerNorm(in_dim, eps=1e-6),
            nn.Linear(in_dim, out_dim),  # reduce dim
            leaky_relu(),
        ]
        for i in range(depth):
            layers.extend([EqualLinear(out_dim, out_dim, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x = F.normalize(x, dim=1)
        return self.net(x)
#*********************************************


from torch.autograd import grad
class WGAN_GP(torch.nn.Module):
    def __init__(self, in_dim: int = 3 * 512):
        super(WGAN_GP, self).__init__()
        # self.discriminator = nn.Sequential(
        #     nn.Linear(in_dim, in_dim),
        #     nn.GELU(),
        #     nn.Linear(in_dim, in_dim // 2),
        #     nn.GELU(),
        #     nn.Linear(in_dim // 2, 1),
        # )
        reduce_scale = 1
        self.discriminator = nn.Sequential(
            StyleVectorizer(in_dim, in_dim // reduce_scale, depth=1, lr_mul=5),
            nn.Linear(in_dim // reduce_scale, 1),
        )
        self.gamma = 10
        self.wd_clf = 1

        self.cache_wdist = 0.
        self.cache_gp = 0.

    def forward(self, target_feat, source_feat=None, is_discriminator_batch: bool = True, gt_labels=None,):
        if source_feat is None:
            assert is_discriminator_batch, "source_feat should be given when is_discriminator_batch=True"
            source_feat = target_feat
        if source_feat.shape[0] > target_feat.shape[0]:
            source_feat = source_feat[:target_feat.shape[0]]
        elif target_feat.shape[0] > source_feat.shape[0]:
            target_feat = target_feat[:source_feat.shape[0]]
        bs = source_feat.shape[0]
        source_feat = source_feat.view(bs, -1)
        target_feat = target_feat.view(bs, -1)
        device = source_feat.device

        if is_discriminator_batch:
            self.cache_gp = gp = self.gradient_penalty(self.discriminator, source_feat, target_feat, device)
            d_source = self.discriminator(source_feat)
            d_target = self.discriminator(target_feat)
            self.cache_wdist = wasserstein_distance = d_source.mean() - d_target.mean()
            critic_cost = -wasserstein_distance + self.gamma * gp
            loss = critic_cost
        else:
            d_target = self.discriminator(target_feat)  # large:real
            d_target_neg_logit = -d_target.mean()
            loss = self.wd_clf * d_target_neg_logit
        return {
            'loss': loss,
            'w_dist': self.cache_wdist,
            'gp': self.cache_gp,
        }

    def gradient_penalty(self, critic, h_s, h_t, device):
        # based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
        alpha = torch.rand(h_s.size(0), 1).to(device)
        differences = h_t - h_s
        interpolates = h_s + (alpha * differences)
        interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()

        preds = critic(interpolates)
        gradients = grad(preds, interpolates,
                         grad_outputs=torch.ones_like(preds),
                         retain_graph=True, create_graph=True)[0]
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        return gradient_penalty
