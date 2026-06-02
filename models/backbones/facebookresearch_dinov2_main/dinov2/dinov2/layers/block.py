# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from typing import Callable, List, Any, Tuple, Dict

import torch
from torch import nn, Tensor

from .attention import Attention, MemEffAttention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp

logger = logging.getLogger("dinov2")

try:
    from xformers.ops import fmha
    from xformers.ops import scaled_index_add, index_select_cat

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False


# =============================================================================
# --- Convpass Adapter Implementation for DINOv2 ---
# =============================================================================
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Convpass(nn.Module):
    def __init__(self, in_dim=768, dim=8, xavier_init=False):
        super().__init__()
        self.adapter_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.weight)
        else:
            nn.init.zeros_(self.adapter_conv.weight)
            self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(dim, dtype=torch.float)
        nn.init.zeros_(self.adapter_conv.bias)

        self.adapter_down = nn.Linear(in_dim, dim)
        self.adapter_up = nn.Linear(dim, in_dim)
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape

        # 智能推导 H 和 W (自动处理 DINOv2 包含 1个 cls 或 多包含 4个 register tokens 的情况)
        grid_size = int(math.isqrt(N))
        if grid_size * grid_size == N:
            num_extra = 0
        else:
            for num_extra in range(1, 10):
                grid_size = int(math.isqrt(N - num_extra))
                if grid_size * grid_size == N - num_extra:
                    break
            else:
                raise ValueError(f"Cannot deduce spatial grid size for sequence length N={N}")

        H = W = grid_size

        x_down = self.adapter_down(x)
        x_down = self.act(x_down)

        x_extra = x_down[:, :num_extra]  # CLS and/or Register tokens
        x_patch = x_down[:, num_extra:]  # Patch tokens

        # 处理二维 Patch 空间特征
        if x_patch.shape[1] > 0:
            x_patch = x_patch.reshape(B, H, W, self.dim).permute(0, 3, 1, 2)
            x_patch = self.adapter_conv(x_patch)
            x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, H * W, self.dim)

        # 处理额外 Token (模拟 1x1 卷积操作特征融合)
        if num_extra > 0:
            x_extra = x_extra.reshape(B, num_extra, 1, self.dim).permute(0, 3, 1, 2)
            x_extra = self.adapter_conv(x_extra)
            x_extra = x_extra.permute(0, 2, 3, 1).reshape(B, num_extra, self.dim)
            x_down = torch.cat([x_extra, x_patch], dim=1)
        else:
            x_down = x_patch

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        return x_up


def inject_convpass(model, method='convpass', dim=8, s=1.0, xavier_init=False):
    """
    暴露给外部的快捷注入函数。遍历并给所有 Block 加上 Convpass 适配器。
    """
    for _, module in model.named_modules():
        if isinstance(module, Block):
            in_dim = module.norm1.weight.shape[0]
            module.adapter_attn = Convpass(in_dim=in_dim, dim=dim, xavier_init=xavier_init)
            module.s = s
            if method == 'convpass':
                module.adapter_mlp = Convpass(in_dim=in_dim, dim=dim, xavier_init=xavier_init)
            else:
                module.adapter_mlp = None
    return model


# =============================================================================


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = False,
            proj_bias: bool = True,
            ffn_bias: bool = True,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            init_values=None,
            drop_path: float = 0.0,
            act_layer: Callable[..., nn.Module] = nn.GELU,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
            attn_class: Callable[..., nn.Module] = Attention,
            ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

        # --- 为 Convpass 适配器预留坑位 ---
        self.adapter_attn = None
        self.adapter_mlp = None
        self.s = 1.0

    def forward(self, x: Tensor) -> Tensor:
        # 巧妙利用残差函数，让 Convpass 被 DropPath (Stochastic Depth) 完美包裹
        def attn_residual_func(x: Tensor) -> Tensor:
            attn_out = self.ls1(self.attn(self.norm1(x)))
            if getattr(self, 'adapter_attn', None) is not None:
                attn_out = attn_out + self.adapter_attn(self.norm1(x)) * self.s
            return attn_out

        def ffn_residual_func(x: Tensor) -> Tensor:
            ffn_out = self.ls2(self.mlp(self.norm2(x)))
            if getattr(self, 'adapter_mlp', None) is not None:
                ffn_out = ffn_out + self.adapter_mlp(self.norm2(x)) * self.s
            return ffn_out

        if self.training and self.sample_drop_ratio > 0.1:
            x = drop_add_residual_stochastic_depth(
                x, residual_func=attn_residual_func, sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x, residual_func=ffn_residual_func, sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


def drop_add_residual_stochastic_depth(
        x: Tensor,
        residual_func: Callable[[Tensor], Tensor],
        sample_drop_ratio: float = 0.0,
) -> Tensor:
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]
    residual = residual_func(x_subset)
    x_flat = x.flatten(1)
    residual = residual.flatten(1)
    residual_scale_factor = b / sample_subset_size
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
        x_list: List[Tensor],
        residual_func: Callable[[Tensor, Any], Tensor],
        sample_drop_ratio: float = 0.0,
        scaling_vector=None,
) -> Tensor:
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:
            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                attn_out = self.attn(self.norm1(x), attn_bias=attn_bias)
                if getattr(self, 'adapter_attn', None) is not None:
                    attn_out = attn_out + self.adapter_attn(self.norm1(x)) * self.s
                return attn_out

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                ffn_out = self.mlp(self.norm2(x))
                if getattr(self, 'adapter_mlp', None) is not None:
                    ffn_out = ffn_out + self.adapter_mlp(self.norm2(x)) * self.s
                return ffn_out

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls2.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            return x_list
        else:
            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                attn_out = self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))
                if getattr(self, 'adapter_attn', None) is not None:
                    attn_out = attn_out + self.adapter_attn(self.norm1(x)) * self.s
                return attn_out

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                ffn_out = self.ls2(self.mlp(self.norm2(x)))
                if getattr(self, 'adapter_mlp', None) is not None:
                    ffn_out = ffn_out + self.adapter_mlp(self.norm2(x)) * self.s
                return ffn_out

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            assert XFORMERS_AVAILABLE, "Please install xFormers for nested tensors usage"
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError