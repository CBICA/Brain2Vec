# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from typing import Any

import timm.models.vision_transformer
import torch
import torch.nn as nn

from NiChart_ViT.models.utils import PatchEmbedding, off_diagonal


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(self, global_pool: bool = False, **kwargs: Any) -> None:
        super(VisionTransformer, self).__init__(**kwargs)

        img_size = 128 if "img_size" not in kwargs else kwargs["img_size"]
        patch_size = 8 if "patch_size" not in kwargs else kwargs["patch_size"]
        embed_dim = 512 if "embed_dim" not in kwargs else kwargs["embed_dim"]

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            num_hidden=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=False,
        )

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x: torch.Tensor, attn_mask: Any = None) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


class BarlowTwins(nn.Module):

    def __init__(
        self,
        vit_model: nn.Module,
        projector: str,
        _lambda: float,
        eff_batch_size: int,
        **kwargs: Any,
    ) -> None:

        super(BarlowTwins, self).__init__()
        self.vit_backbone = vit_model
        self.projector_dims = list(map(int, projector.split("-")))
        self._lambda = _lambda
        self.eff_batch_size = eff_batch_size

        embed_dim = vit_model.embed_dim
        in_dim = embed_dim

        projector_layers = []
        for i, out_dim in enumerate(self.projector_dims):
            projector_layers.append(nn.Linear(in_dim, out_dim, bias=False))
            projector_layers.append(nn.BatchNorm1d(out_dim))
            if i != len(self.projector_dims) - 1:  # dont add a final ReLU
                projector_layers.append(nn.ReLU(inplace=True))
            in_dim = out_dim

        self.projector = nn.Sequential(*projector_layers)
        self.projector_bn = nn.BatchNorm1d(self.projector_dims[-1], affine=False)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        z1 = self.projector(self.vit_backbone.forward_features(x1))
        z2 = self.projector(self.vit_backbone.forward_features(x2))

        c = self.projector_bn(z1).T @ self.projector_bn(z2)

        c.div_(self.eff_batch_size)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        bt_loss = on_diag + self._lambda * off_diag

        return bt_loss


def vit_base_patch8(**kwargs: Any) -> nn.Module:
    model = VisionTransformer(
        patch_size=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large_patch8(**kwargs: Any) -> nn.Module:
    model = VisionTransformer(
        patch_size=8,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge_patch8(**kwargs: Any) -> nn.Module:
    model = VisionTransformer(
        patch_size=8,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def barlow_twins_base(**kwargs: Any) -> nn.Module:
    img_size = kwargs.get("img_size", 128)
    model = BarlowTwins(
        vit_model=vit_base_patch8(img_size=img_size),
        **kwargs,
    )

    return model


def barlow_twins_large(**kwargs: Any) -> nn.Module:
    img_size = kwargs.get("img_size", 128)
    model = BarlowTwins(
        vit_model=vit_large_patch8(img_size=img_size),
        **kwargs,
    )

    return model


def barlow_twins_huge(**kwargs: Any) -> nn.Module:
    img_size = kwargs.get("img_size", 128)
    model = BarlowTwins(
        vit_model=vit_huge_patch8(img_size=img_size),
        **kwargs,
    )

    return model
