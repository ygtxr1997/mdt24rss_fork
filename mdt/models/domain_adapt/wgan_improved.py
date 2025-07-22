import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import einops
from torch.autograd import grad


class SpectralNorm(nn.Module):
    """可选的谱归一化包装器"""

    def __init__(self, module):
        super().__init__()
        self.module = spectral_norm(module)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class LeakyReLUBlock(nn.Module):
    """标准的LeakyReLU块"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 use_bn=False, use_spectral_norm=True, conv_type='1d'):
        super().__init__()

        # 选择卷积类型
        conv_layer = nn.Conv1d if conv_type == '1d' else nn.Conv2d
        bn_layer = nn.BatchNorm1d if conv_type == '1d' else nn.BatchNorm2d

        conv = conv_layer(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn)

        # 应用谱归一化
        if use_spectral_norm:
            conv = spectral_norm(conv)

        self.conv = conv
        self.bn = bn_layer(out_channels) if use_bn else None
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.activation(x)


class AdaLNZero(nn.Module):
    """改进的AdaLN，支持零初始化"""

    def __init__(self, hidden_size, cond_dim=512):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

        # 使用更稳定的条件映射
        self.modulation = nn.Sequential(
            nn.Linear(cond_dim, hidden_size * 2),  # scale and shift
            nn.Tanh()  # 防止梯度爆炸
        )

        # 零初始化
        nn.init.zeros_(self.modulation[0].weight)
        nn.init.zeros_(self.modulation[0].bias)

    def forward(self, x, condition):
        # x: (..., hidden_size), condition: (batch, cond_dim)
        x_norm = self.norm(x)
        scale_shift = self.modulation(condition)

        # 为了支持不同维度的x，需要正确broadcast
        while scale_shift.ndim < x_norm.ndim:
            scale_shift = scale_shift.unsqueeze(-2)

        scale, shift = scale_shift.chunk(2, dim=-1)
        return x_norm * (1 + scale) + shift


class Discriminator1dImproved(nn.Module):
    """改进的1D判别器，适用于特征向量判别"""

    def __init__(self,
                 in_dim: int,
                 inner_dim: int = 64,
                 num_layers: int = 4,
                 use_spectral_norm: bool = True,
                 use_self_attention: bool = False,
                 use_adaptive_norm: bool = False,
                 condition_dim: int = 512,
                 dropout: float = 0.0,
                 final_activation: str = 'none'  # 'none', 'sigmoid', 'tanh'
                 ):
        super().__init__()

        self.use_adaptive_norm = use_adaptive_norm
        self.use_self_attention = use_self_attention

        # 计算合适的下采样率
        total_downsample = 4 ** (num_layers - 1)
        if in_dim < total_downsample * 8:
            # 对于小维度，减少下采样
            strides = [2] * (num_layers - 1) + [1]
        else:
            strides = [4] * (num_layers - 1) + [1]

        # 构建网络层
        layers = []
        current_dim = 1  # 输入通道数

        for i in range(num_layers):
            out_dim = inner_dim * (2 ** min(i, 3))  # 限制最大通道数
            stride = strides[i] if i < len(strides) else 1

            layers.append(LeakyReLUBlock(
                current_dim, out_dim,
                kernel_size=4, stride=stride, padding=1,
                use_bn=False,  # WGAN-GP不推荐使用BN
                use_spectral_norm=use_spectral_norm,
                conv_type='1d'
            ))

            # 自注意力机制（在中间层添加）
            if use_self_attention and i == num_layers // 2:
                layers.append(SelfAttention1D(out_dim))

            current_dim = out_dim

        self.feature_extractor = nn.Sequential(*layers)

        # 自适应归一化
        if use_adaptive_norm:
            self.ada_norms = nn.ModuleList([
                AdaLNZero(inner_dim * (2 ** min(i, 3)), condition_dim)
                for i in range(num_layers)
            ])

        # 计算输出维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, in_dim)
            dummy_output = self.feature_extractor(dummy_input)
            dummy_output = F.adaptive_avg_pool1d(dummy_output, 1).squeeze(-1)  # (B, C)
            output_size = dummy_output.view(dummy_output.size(0), -1).size(1)

        # 最终分类器
        classifier = [nn.Dropout(dropout)] if dropout > 0 else []

        final_linear = nn.Linear(output_size, 1)
        if use_spectral_norm:
            final_linear = spectral_norm(final_linear)
        classifier.append(final_linear)

        # 最终激活函数
        if final_activation == 'sigmoid':
            classifier.append(nn.Sigmoid())
        elif final_activation == 'tanh':
            classifier.append(nn.Tanh())

        self.classifier = nn.Sequential(*classifier)

        # 参数初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            # 使用正交初始化，对WGAN更稳定
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, condition=None):
        """
        Args:
            x: (B, D) or (B, 1, D) or (B, T, D)
            condition: (B, condition_dim) 可选的条件信息
        """
        # 输入预处理
        if x.ndim == 2:  # (B, D)
            x = x.unsqueeze(1)  # (B, 1, D)
        elif x.ndim == 3:  # (B, T, D)
            if x.size(1) > 1:  # 如果有时间维度，进行池化
                x = torch.mean(x, dim=1, keepdim=True)  # (B, 1, D)

        # 前向传播
        if self.use_adaptive_norm:
            # 逐层应用自适应归一化
            for i, (layer, norm) in enumerate(zip(self.feature_extractor, self.ada_norms)):
                x = layer(x)
                if condition is not None:
                    # 转换为适合的形状进行归一化
                    x_reshaped = x.transpose(1, 2)  # (B, D, C) -> (B, C, D)
                    x_normalized = norm(x_reshaped, condition)
                    x = x_normalized.transpose(1, 2)  # (B, C, D) -> (B, D, C)
        else:
            x = self.feature_extractor(x)

        # 全局池化和分类
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (B, C)
        return self.classifier(x)

    def calc_params(self):
        num_params = sum(p.numel() for p in self.parameters())
        ret_str = f"{num_params/1024/1024:.2f}M Params"
        return ret_str


class SelfAttention1D(nn.Module):
    """1D自注意力机制"""

    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv1d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv1d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv1d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, W = x.size()

        query = self.query_conv(x).view(B, -1, W).permute(0, 2, 1)  # (B, W, C//8)
        key = self.key_conv(x).view(B, -1, W)  # (B, C//8, W)
        value = self.value_conv(x).view(B, -1, W)  # (B, C, W)

        attention = torch.bmm(query, key)  # (B, W, W)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, W)
        out = self.gamma * out + x

        return out


class Discriminator2dImproved(nn.Module):
    """改进的2D判别器，适用于序列数据"""

    def __init__(self,
                 in_dim: int,
                 time_dim: int = 10,
                 inner_dim: int = 64,
                 num_layers: int = 4,
                 use_spectral_norm: bool = True,
                 use_self_attention: bool = False,
                 use_adaptive_norm: bool = False,
                 condition_dim: int = 512,
                 dropout: float = 0.0,
                 final_activation: str = 'none'
                 ):
        super().__init__()

        self.use_adaptive_norm = use_adaptive_norm

        # 网络结构设计
        layers = []
        current_channels = 1

        # 第一层：处理时间和特征维度
        layers.append(LeakyReLUBlock(
            current_channels, inner_dim,
            kernel_size=(3, 4), stride=(1, 4), padding=(1, 1),
            use_spectral_norm=use_spectral_norm,
            conv_type='2d'
        ))
        current_channels = inner_dim

        # 中间层：逐步下采样
        for i in range(1, num_layers):
            out_channels = inner_dim * (2 ** min(i, 3))

            # 根据层数调整stride
            if i == 1:
                stride = (2, 4)
            elif i == 2:
                stride = (2, 4)
            else:
                stride = (1, 2)

            layers.append(LeakyReLUBlock(
                current_channels, out_channels,
                kernel_size=(3, 4), stride=stride, padding=(1, 1),
                use_spectral_norm=use_spectral_norm,
                conv_type='2d'
            ))

            # 在中间层添加自注意力
            if use_self_attention and i == num_layers // 2:
                layers.append(SelfAttention2D(out_channels))

            current_channels = out_channels

        self.feature_extractor = nn.Sequential(*layers)

        # 自适应归一化
        if use_adaptive_norm:
            self.ada_norms = nn.ModuleList([
                AdaLNZero(inner_dim * (2 ** min(i, 3)), condition_dim)
                for i in range(num_layers)
            ])

        # 计算输出维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, time_dim, in_dim)
            dummy_output = self.feature_extractor(dummy_input)
            dummy_output = F.adaptive_avg_pool2d(dummy_output, (1, 1)).squeeze(-1).squeeze(-1)
            output_size = dummy_output.view(dummy_output.size(0), -1).size(1)

        # 分类器
        classifier = [nn.Dropout(dropout)] if dropout > 0 else []

        final_linear = nn.Linear(output_size, 1)
        if use_spectral_norm:
            final_linear = spectral_norm(final_linear)
        classifier.append(final_linear)

        if final_activation == 'sigmoid':
            classifier.append(nn.Sigmoid())
        elif final_activation == 'tanh':
            classifier.append(nn.Tanh())

        self.classifier = nn.Sequential(*classifier)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, condition=None):
        """
        Args:
            x: (B, T, D) 序列数据
            condition: (B, condition_dim) 可选条件
        """
        if x.ndim == 3:  # (B, T, D)
            x = x.unsqueeze(1)  # (B, 1, T, D)

        # 特征提取
        if self.use_adaptive_norm:
            for i, (layer, norm) in enumerate(zip(self.feature_extractor, self.ada_norms)):
                x = layer(x)
                if condition is not None:
                    # 对2D特征图应用归一化
                    B, C, H, W = x.shape
                    x_flat = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
                    x_norm = norm(x_flat, condition)
                    x = x_norm.permute(0, 2, 1).view(B, C, H, W)
        else:
            x = self.feature_extractor(x)

        # 全局池化和分类
        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        return self.classifier(x)


class SelfAttention2D(nn.Module):
    """2D自注意力机制"""

    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()

        query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        key = self.key_conv(x).view(B, -1, H * W)
        value = self.value_conv(x).view(B, -1, H * W)

        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        out = self.gamma * out + x

        return out


class WGAN_GP_Improved(nn.Module):
    """改进的WGAN-GP实现，兼容原有接口"""

    def __init__(self,
                 in_dim: str = "1536*6,",
                 in_ndim: str = "2*6,",
                 time_dim: int = 10,
                 inner_dim: int = 64,
                 gamma: float = 10,  # 保持原有参数名
                 num_layers: int = 1,
                 use_ada: bool = False,
                 sigma_dim: int = 512,
                 use_cond_dist: bool = False,
                 use_bn: bool = False,  # 改进版默认不使用BN
                 # 新增的改进参数
                 use_spectral_norm: bool = True,
                 use_self_attention: bool = False,
                 lambda_drift: float = 0.001,
                 use_r1_regularization: bool = False,
                 lambda_r1: float = 10.0,
                 improved_discriminator: bool = True  # 是否使用改进版判别器
                 ):
        super().__init__()

        self.num_layers = num_layers
        self.gamma = gamma  # 梯度惩罚系数
        self.wd_clf = 1
        self.use_cond_dist = use_cond_dist
        self.improved_discriminator = improved_discriminator

        # 解析维度配置
        in_dims = self.process_in_dim_str(in_dim)
        in_ndims = self.process_in_dim_str(in_ndim)
        assert len(in_dims) == num_layers

        # 构建判别器
        discriminators = []
        for l in range(self.num_layers):
            if improved_discriminator:
                # 使用改进的判别器
                d_net = self.get_improved_discriminators(
                    in_ndims[l],
                    in_dim=in_dims[l],
                    inner_dim=inner_dim,
                    use_ada=use_ada,
                    use_cond_dist=use_cond_dist,
                    time_dim=time_dim,
                    sigma_dim=sigma_dim,
                    use_spectral_norm=use_spectral_norm,
                    use_self_attention=use_self_attention
                )
            else:
                # 使用原有的判别器
                d_net = self.get_discriminators(
                    in_ndims[l],
                    in_dim=in_dims[l],
                    inner_dim=inner_dim,
                    use_ada=use_ada,
                    use_cond_dist=use_cond_dist,
                    time_dim=time_dim,
                    use_bn=use_bn,
                    sigma_dim=sigma_dim,
                )
            discriminators.append(d_net)

        self.discriminators = nn.ModuleList(discriminators)

        # 条件距离映射
        if self.use_cond_dist:
            self.cond_dist_mapping = CondDistMapping()

        # 改进参数
        self.lambda_drift = lambda_drift
        self.lambda_r1 = lambda_r1
        self.use_r1_regularization = use_r1_regularization

        # 缓存指标
        self.cache_wdists = [0. for _ in range(self.num_layers)]
        self.cache_gps = [0. for _ in range(self.num_layers)]

    @staticmethod
    def process_in_dim_str(in_dim: str):
        """保持原有的维度解析逻辑"""
        discriminators = in_dim.split(',')
        dims_list = []
        for discriminator in discriminators:
            if discriminator == '': continue
            dim, layer = [int(x) for x in discriminator.split('*')]
            dims_list.extend([dim] * layer)
        return dims_list

    @staticmethod
    def get_discriminators(ndim: int, **kwargs):
        """原有的判别器构建方法"""
        if ndim == 2:  # (B,D)
            return Discriminator1d(ndim=ndim, **kwargs)
        elif ndim == 3:  # (B,T,D)
            return Discriminator2d(ndim=ndim, **kwargs)
        elif ndim == 4:  # (B,Nh,T,Tc)
            return Discriminator3d(ndim=ndim, **kwargs)
        else:
            raise NotImplementedError(f"{ndim} not supported!")

    @staticmethod
    def get_improved_discriminators(ndim: int, **kwargs):
        """新的改进判别器构建方法"""
        # 提取改进版参数
        improved_params = {
            'use_spectral_norm': kwargs.pop('use_spectral_norm', True),
            'use_self_attention': kwargs.pop('use_self_attention', False),
            'use_adaptive_norm': kwargs.get('use_ada', False),
            'condition_dim': kwargs.get('sigma_dim', 512),
            'dropout': 0.1
        }

        if ndim == 2:  # (B,D)
            return Discriminator1dImproved(
                in_dim=kwargs['in_dim'],
                inner_dim=kwargs.get('inner_dim', 64),
                **improved_params
            )
        elif ndim == 3:  # (B,T,D)
            return Discriminator2dImproved(
                in_dim=kwargs['in_dim'],
                time_dim=kwargs.get('time_dim', 10),
                inner_dim=kwargs.get('inner_dim', 64),
                **improved_params
            )
        else:
            # 对于4D等复杂情况，回退到原有实现
            return Discriminator3d(ndim=ndim, **kwargs)

    def forward(self, target_feats, source_feats=None, is_discriminator_batch: bool = True,
                sigmas: torch.Tensor = None,
                conditions: list = None,  # 保持原有参数
                ):
        """保持与原有forward接口完全兼容"""
        if source_feats is None:
            assert not is_discriminator_batch, "source_feat should be given when is_discriminator_batch=True"
            source_feats = target_feats

        assert len(target_feats) == len(source_feats) == self.num_layers

        # 条件距离计算（保持原有逻辑）
        if self.use_cond_dist:
            assert len(conditions) == 2
            conditions = [cond.reshape(cond.shape[0], -1) for cond in conditions]
            dist_emb = self.cond_dist_mapping(conditions[0], conditions[1])
            sigmas = torch.cat([sigmas, dist_emb], dim=-1)

        loss = 0.
        for l in range(self.num_layers):
            layer_discriminator = self.discriminators[l]
            target_feat = target_feats[l]
            source_feat = source_feats[l]

            # 处理批量大小不匹配（保持原有逻辑）
            if source_feat.shape[0] > target_feat.shape[0]:
                source_feat = source_feat[-target_feat.shape[0]:]
                print('[Warning] target < source feat')
            elif target_feat.shape[0] > source_feat.shape[0]:
                target_feat = target_feat[-source_feat.shape[0]:]
                print('[Warning] target > source feat')

            bs = source_feat.shape[0]
            device = source_feat.device

            if is_discriminator_batch:
                # 判别器训练
                if self.improved_discriminator:
                    # 使用改进的梯度惩罚
                    self.cache_gps[l] = gp = self.improved_gradient_penalty(
                        layer_discriminator, source_feat, target_feat, sigmas
                    )
                else:
                    # 使用原有的梯度惩罚
                    self.cache_gps[l] = gp = self.gradient_penalty(
                        layer_discriminator, source_feat, target_feat, device, sigmas
                    )

                d_source = layer_discriminator(source_feat.detach(), sigmas)
                d_target = layer_discriminator(target_feat.clone().detach(), sigmas)
                self.cache_wdists[l] = wasserstein_distance = d_source.mean() - d_target.mean()

                # 基础损失
                critic_cost = -wasserstein_distance + self.gamma * gp

                # 添加改进的正则化项
                if self.improved_discriminator:
                    # Drift penalty
                    if self.lambda_drift > 0:
                        drift_penalty = (d_source ** 2).mean()
                        critic_cost += self.lambda_drift * drift_penalty

                    # R1正则化
                    if self.use_r1_regularization:
                        r1_penalty = self.compute_r1_penalty(layer_discriminator, source_feat, sigmas)
                        critic_cost += self.lambda_r1 * r1_penalty

                loss += critic_cost
            else:
                # 生成器训练
                d_source = layer_discriminator(source_feat.detach(), sigmas)
                d_target = layer_discriminator(target_feat, sigmas)
                d_target_neg_logit = d_source.mean() - d_target.mean()
                loss += self.wd_clf * d_target_neg_logit

        loss = loss / self.num_layers

        # 返回与原有接口兼容的格式
        return {
            'loss': loss,
            'w_dist': sum(self.cache_wdists) / self.num_layers,
            'gp': sum(self.cache_gps) / self.num_layers,
        }

    def gradient_penalty(self, critic, h_s, h_t, device, sigmas):
        """原有的梯度惩罚实现，保持兼容"""
        alpha = torch.rand(h_s.size(0)).to(device)
        while alpha.ndim < h_s.ndim:
            alpha = alpha.unsqueeze(-1)
        differences = h_t - h_s
        interpolates = h_s + (alpha * differences)
        interpolates.requires_grad_(True)

        preds = critic(interpolates, sigmas)

        gradients = grad(preds, interpolates,
                         grad_outputs=torch.ones_like(preds),
                         retain_graph=True, create_graph=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        return gradient_penalty

    def improved_gradient_penalty(self, discriminator, real_data, fake_data, condition=None):
        """改进的梯度惩罚实现"""
        batch_size = real_data.size(0)
        device = real_data.device

        alpha = torch.rand(batch_size, device=device)
        for _ in range(real_data.ndim - 1):
            alpha = alpha.unsqueeze(-1)

        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)

        d_interpolated = discriminator(interpolated, condition)

        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients_flat = gradients.view(batch_size, -1)
        gradient_norm = gradients_flat.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()

        return gradient_penalty

    def compute_r1_penalty(self, discriminator, real_data, condition=None):
        """R1正则化"""
        real_data = real_data.requires_grad_(True)
        d_real = discriminator(real_data, condition)

        gradients = torch.autograd.grad(
            outputs=d_real.sum(),
            inputs=real_data,
            create_graph=True,
            retain_graph=True
        )[0]

        r1_penalty = gradients.pow(2).sum(dim=tuple(range(1, gradients.ndim))).mean()
        return r1_penalty