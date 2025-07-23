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


class MLPBlock(nn.Module):
    """MLP块，支持谱归一化和Dropout"""

    def __init__(self, in_features, out_features, use_spectral_norm=True, dropout=0.0, activation='leaky_relu'):
        super().__init__()

        linear = nn.Linear(in_features, out_features)

        if use_spectral_norm:
            linear = spectral_norm(linear)

        self.linear = linear
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        x = self.linear(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Conv1DBlock(nn.Module):
    """1D卷积块，用于处理序列数据"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 use_spectral_norm=True, dropout=0.0, activation='leaky_relu'):
        super().__init__()

        conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

        if use_spectral_norm:
            conv = spectral_norm(conv)

        self.conv = conv
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


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
    """基于纯MLP的1D判别器，适用于特征向量判别 (B,D)"""

    def __init__(self,
                 in_dim: int,
                 inner_dim: int = 256,
                 num_layers: int = 4,
                 use_spectral_norm: bool = True,
                 use_adaptive_norm: bool = False,
                 condition_dim: int = 512,
                 dropout: float = 0.1,
                 final_activation: str = 'none'  # 'none', 'sigmoid', 'tanh'
                 ):
        super().__init__()

        self.use_adaptive_norm = use_adaptive_norm
        self.num_layers = num_layers

        # 构建MLP层
        layers = []
        layer_dims = []

        # 计算每层的维度
        current_dim = in_dim
        for i in range(num_layers):
            if i == 0:
                # 第一层可以扩展维度
                out_dim = inner_dim
            elif i == num_layers - 1:
                # 最后一层收缩到较小维度
                out_dim = inner_dim // 4
            else:
                # 中间层逐渐收缩
                out_dim = inner_dim // (2 ** (i - 1))

            layer_dims.append(out_dim)

            # 添加MLP块
            layers.append(MLPBlock(
                current_dim, out_dim,
                use_spectral_norm=use_spectral_norm,
                dropout=dropout if i > 0 else 0.0,  # 第一层不使用dropout
                activation='leaky_relu'
            ))

            current_dim = out_dim

        self.feature_extractor = nn.Sequential(*layers)

        # 自适应归一化
        if use_adaptive_norm:
            self.ada_norms = nn.ModuleList([
                AdaLNZero(layer_dims[i], condition_dim)
                for i in range(num_layers)
            ])

        # 最终分类器
        classifier = []
        if dropout > 0:
            classifier.append(nn.Dropout(dropout))

        final_linear = nn.Linear(current_dim, 1)
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
        if isinstance(m, nn.Linear):
            # 使用正交初始化，对WGAN更稳定
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, condition=None):
        """
        Args:
            x: (B, D) 特征向量
            condition: (B, condition_dim) 可选的条件信息
        """
        # 输入预处理 - 确保是2D张量
        if x.ndim == 3 and x.size(1) == 1:  # (B, 1, D)
            x = x.squeeze(1)  # (B, D)
        elif x.ndim == 3:  # (B, T, D) - 对时间维度求平均
            x = x.mean(dim=1)  # (B, D)

        assert x.ndim == 2, f"Expected 2D input (B, D), got shape {x.shape}"

        # 前向传播
        if self.use_adaptive_norm and condition is not None:
            # 逐层应用自适应归一化
            for i, (layer, norm) in enumerate(zip(self.feature_extractor, self.ada_norms)):
                x = layer(x)
                x = norm(x, condition)
        else:
            x = self.feature_extractor(x)

        # 分类
        return self.classifier(x)

    def calc_params(self):
        num_params = sum(p.numel() for p in self.parameters())
        ret_str = f"{num_params / 1024 / 1024:.2f}M Params"
        return ret_str


class Discriminator2dImproved(nn.Module):
    """基于1D卷积的2D判别器，适用于序列数据 (B,T,D)"""

    def __init__(self,
                 in_dim: int,  # 特征维度D
                 time_dim: int = 10,  # 时间维度T（用于参数计算）
                 inner_dim: int = 128,
                 num_layers: int = 4,
                 use_spectral_norm: bool = True,
                 use_adaptive_norm: bool = False,
                 condition_dim: int = 512,
                 dropout: float = 0.1,
                 final_activation: str = 'none'
                 ):
        super().__init__()

        self.use_adaptive_norm = use_adaptive_norm
        self.in_dim = in_dim
        self.time_dim = time_dim

        # 根据time_dim智能调整层数，避免过度下采样
        max_possible_layers = 0
        temp_time = time_dim
        while temp_time > 4:  # 至少保留4个时间步
            temp_time = (temp_time + 2 * 1 - 3) // 2 + 1  # Conv1d output size计算
            max_possible_layers += 1

        # 调整实际使用的层数
        effective_num_layers = min(num_layers, max_possible_layers, 6)  # 最多6层
        self.num_layers = effective_num_layers

        print(
            f"[Info] Discriminator2dImproved: time_dim={time_dim}, requested_layers={num_layers}, effective_layers={effective_num_layers}")

        # 简洁的1D卷积层设计：所有层都沿时间维度做卷积
        layers = []
        layer_dims = []

        # 输入通道数是特征维度D
        current_channels = in_dim

        for i in range(effective_num_layers):
            # 统一的卷积层设计
            out_channels = inner_dim * (2 ** min(i, 3))  # 通道数逐层增加，最多8倍

            # 自适应的1D卷积参数
            kernel_size = 3  # 统一使用3，更安全
            stride = 2 if i < effective_num_layers - 1 else 1  # 最后一层不下采样
            padding = 1

            layer_dims.append(out_channels)

            layers.append(Conv1DBlock(
                current_channels, out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
                use_spectral_norm=use_spectral_norm,
                dropout=dropout if i > 0 else 0.0,
                activation='leaky_relu'
            ))

            current_channels = out_channels

        self.feature_extractor = nn.Sequential(*layers)

        # 自适应归一化
        if use_adaptive_norm:
            self.ada_norms = nn.ModuleList([
                AdaLNZero(layer_dims[i], condition_dim)
                for i in range(effective_num_layers)
            ])

        # 计算输出维度（通过实际前向传播）
        with torch.no_grad():
            dummy_input = torch.randn(1, time_dim, in_dim)
            dummy_processed = self._preprocess_input(dummy_input)
            dummy_output = self.feature_extractor(dummy_processed)

            # 全局平均池化
            dummy_pooled = F.adaptive_avg_pool1d(dummy_output, 1).squeeze(-1)
            output_size = dummy_pooled.size(1)

        # 最终分类器
        classifier = []
        if dropout > 0:
            classifier.append(nn.Dropout(dropout))

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
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _preprocess_input(self, x):
        """
        将(B, T, D)转换为适合1D卷积的格式: (B, D, T)
        简洁的预处理：只做维度转换，让特征维度作为通道数
        """
        # (B, T, D) -> (B, D, T)
        # D作为通道数，T作为序列长度，沿时间维度做卷积
        return x.transpose(1, 2)

    def forward(self, x, condition=None):
        """
        Args:
            x: (B, T, D) 序列数据
            condition: (B, condition_dim) 可选条件
        """
        # 输入预处理
        if x.ndim == 2:  # (B, D) -> 假设T=1
            x = x.unsqueeze(1)  # (B, 1, D)

        assert x.ndim == 3, f"Expected 3D input (B, T, D), got shape {x.shape}"

        # 预处理输入: (B, T, D) -> (B, D, T)
        x = self._preprocess_input(x)

        # 特征提取
        if self.use_adaptive_norm and condition is not None:
            for i, (layer, norm) in enumerate(zip(self.feature_extractor, self.ada_norms)):
                x = layer(x)  # (B, C, T)

                # 对1D特征图应用归一化
                B, C, T = x.shape
                x_reshaped = x.transpose(1, 2)  # (B, T, C)
                x_norm = norm(x_reshaped, condition)  # (B, T, C)
                x = x_norm.transpose(1, 2)  # (B, C, T)
        else:
            x = self.feature_extractor(x)

        # 全局平均池化
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (B, C)

        # 分类
        return self.classifier(x)

    def calc_params(self):
        num_params = sum(p.numel() for p in self.parameters())
        ret_str = f"{num_params / 1024 / 1024:.2f}M Params"
        return ret_str


class SelfAttention1D(nn.Module):
    """1D自注意力机制（保留原有实现）"""

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
    def get_improved_discriminators(ndim: int, **kwargs):
        """新的改进判别器构建方法"""
        # 提取改进版参数
        improved_params = {
            'use_spectral_norm': kwargs.pop('use_spectral_norm', True),
            'use_adaptive_norm': kwargs.get('use_ada', False),
            'condition_dim': kwargs.get('sigma_dim', 512),
            'dropout': 0.1
        }

        if ndim == 2:  # (B,D) - 使用MLP判别器
            return Discriminator1dImproved(
                in_dim=kwargs['in_dim'],
                inner_dim=kwargs.get('inner_dim', 256),
                **improved_params
            )
        elif ndim == 3:  # (B,T,D) - 使用1D卷积判别器
            return Discriminator2dImproved(
                in_dim=kwargs['in_dim'],
                time_dim=kwargs.get('time_dim', 10),
                inner_dim=kwargs.get('inner_dim', 128),
                **improved_params
            )
        else:
            # 对于4D等复杂情况，需要另外实现
            raise NotImplementedError(f"ndim={ndim} not implemented in improved version")

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
                # 使用改进的梯度惩罚
                self.cache_gps[l] = gp = self.improved_gradient_penalty(
                    layer_discriminator, source_feat, target_feat, sigmas
                )

                d_source = layer_discriminator(source_feat.detach(), sigmas)
                d_target = layer_discriminator(target_feat.clone().detach(), sigmas)
                self.cache_wdists[l] = wasserstein_distance = d_source.mean() - d_target.mean()

                # 基础损失
                critic_cost = -wasserstein_distance + self.gamma * gp

                # 添加改进的正则化项
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

        # print(interpolated.shape)

        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients_flat = gradients.reshape(batch_size, -1)
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


# 为了兼容性，需要实现CondDistMapping类（如果原代码中有的话）
class CondDistMapping(nn.Module):
    """条件距离映射（示例实现）"""

    def __init__(self, input_dim=512, output_dim=64):
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, cond1, cond2):
        # 计算两个条件的距离特征
        combined = torch.cat([cond1, cond2], dim=-1)
        return self.mapping(combined)