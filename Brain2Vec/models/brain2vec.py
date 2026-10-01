# Copyright 2026 CBICA
#
# Author: Spiros Maggioros
#
# References:
#   facebookresearch: https://github.com/facebookresearch/mae/blob/main/models_mae.py
#   timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
#   DeiT: https://github.com/facebookresearch/deit
#
# Licenced under CBICA's non-Commercial Software Agreement:
# https://www.med.upenn.edu/cbica/software-agreement-non-commercial.html


from functools import partial
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from Brain2Vec.models.utils import PatchEmbedding, get_3d_sincos_pos_embed


class Brain2Vec(nn.Module):
    """
    Brain2Vec class implementation

    :param img_size: Size of 3D input, most of the times = 128
    :type img_size: int
    :param patch_size: Patch size for the vision transformers, we suggest = 8 for img_size = 128
    :type patch_size: int
    :param embed_dim: Size of the encoder's output embeddings
    :type embed_dim: int
    :param depth: Number of ViT blocks for the encoder
    :type depth: int
    :param decoder_embed_dim: Size of the decoder's output embeddings
    :type decoder_embed_dim: int
    :param decoder_depth: Number of ViT blocks for the decoder
    :type decoder_depth: int
    :param mlp_ratio: Expansion of output embeddings (e.g. for mlp_ratio = 4.0, from 768 -> 4.0 * 768)
    :type mlp_ratio: float
    :param norm_layer: The type of normalization layer, we suggest nn.LayerNorm
    :type norm_layer: nn.
    :param norm_pix_loss: Set to true if you only care about representation learning and not voxel-perfect reconstruction
    :type norm_pix_loss: bool
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int = 8,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: Any = nn.LayerNorm,
        norm_pix_loss: bool = False,
    ) -> None:
        super().__init__()

        # Encoder specifics
        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, num_hidden=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**3, bias=True)

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self) -> None:
        num_patches = self.patch_embed.num_patches + 1
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            round(num_patches ** (1 / 3)),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_3d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            round(num_patches ** (1 / 3)),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        w = self.patch_embed.conv.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m: Any) -> None:
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: (N, C, D, H, W)
        x: (N, L, patch_size**3 * C)
        """
        p = self.patch_embed.patch_size[0]
        assert (
            imgs.shape[2] == imgs.shape[3]
            and imgs.shape[3] == imgs.shape[4]
            and imgs.shape[2] % p == 0
        )

        c = imgs.shape[1]
        d = h = w = imgs.shape[2] // p

        x = imgs.reshape(shape=(imgs.shape[0], c, d, p, h, p, w, p))

        # Rearrange dimensions:  (N, C, d, p, h, p, w, p) -> (N, d, h, w, p, p, p, C)
        x = torch.einsum("ncdihpwq->ndhwipqc", x)
        x = x.reshape(shape=(imgs.shape[0], d * h * w, p**3 * c))

        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, L, patch_size**3 * C)
        imgs: (N, C, D, H, W)
        """
        p = self.patch_embed.patch_size[0]
        c = x.shape[2] // (p**3)

        d = h = w = round((x.shape[1]) ** (1 / 3))

        assert d * h * w == x.shape[1]

        # Reshape to separate patch and channel dimensions
        x = x.reshape(shape=(x.shape[0], d, h, w, p, p, p, c))

        # Rearrange dimensions:  (N, d, h, w, p, p, p, C) -> (N, C, d, p, h, p, w, p)
        x = torch.einsum("ndhwipqc->ncdihpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, d * p, h * p, w * p))

        return imgs

    def mask_specific(
        self,
        x: torch.Tensor,
        ground_truth: torch.Tensor,
        mask_ratio: float,
    ) -> Any:
        """
        Mask patches only where ground_truth == 1 (patches outside the ROI,
        ground_truth == 0, are always kept visible).

        mask_ratio here is the fraction of the ROI itself to mask (not a
        fraction of the whole volume). Use mask_ratio=1.0 to mask the
        entire ROI and force the decoder to fully regenerate it.
        """
        N, L, D = x.shape
        device = x.device

        ids_keep_list = []
        ids_shuffle_list = []
        len_keep_list = []

        for i in range(N):
            keep_always = torch.where(ground_truth[i] == 0)[0]

            valid_indices = torch.where(ground_truth[i] == 1)[0]
            num_valid = len(valid_indices)

            len_keep_valid = int(num_valid * (1 - mask_ratio))

            perm = torch.randperm(num_valid, device=device)
            keep_valid = valid_indices[perm[:len_keep_valid]]
            mask_valid = valid_indices[perm[len_keep_valid:]]

            ids_keep = torch.cat([keep_always, keep_valid])
            ids_keep = ids_keep[torch.argsort(ids_keep)]

            ids_shuffle = torch.cat([ids_keep, mask_valid])

            ids_keep_list.append(ids_keep)
            ids_shuffle_list.append(ids_shuffle)
            len_keep_list.append(len(ids_keep))

        len_keep = min(len_keep_list)

        ids_shuffle = torch.stack(
            ids_shuffle_list
        )  # [N, L], full permutation per sample
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def block_masking(self, x: torch.Tensor, mask_ratio: float) -> Any:
        N, L, _ = x.shape
        device = x.device

        grid_size = int(round(L ** (1 / 3)))
        block_size = int(grid_size * mask_ratio)

        block_size = min(block_size, grid_size)
        max_start = grid_size - block_size

        x_masked_list = []
        mask_list = []
        ids_restore_list = []

        for n in range(N):
            start_d = torch.randint(0, max_start + 1, (1,), device=device).item()
            start_h = torch.randint(0, max_start + 1, (1,), device=device).item()
            start_w = torch.randint(0, max_start + 1, (1,), device=device).item()

            mask_3d = torch.zeros(grid_size, grid_size, grid_size, device=device)
            mask_3d[
                start_d : start_d + block_size,
                start_h : start_h + block_size,
                start_w : start_w + block_size,
            ] = 1

            mask_flat = mask_3d.flatten()

            ids_keep = torch.where(mask_flat == 0)[0]
            ids_mask = torch.where(mask_flat == 1)[0]

            ids_shuffle = torch.cat([ids_keep, ids_mask])
            ids_restore = torch.argsort(ids_shuffle)

            x_masked_list.append(x[n : n + 1, ids_keep, :])
            mask_list.append(mask_flat.unsqueeze(0))
            ids_restore_list.append(ids_restore.unsqueeze(0))

        x_masked = torch.cat(x_masked_list, dim=0)
        mask = torch.cat(mask_list, dim=0)
        ids_restore = torch.cat(ids_restore_list, dim=0)

        return x_masked, mask, ids_restore

    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> Any:
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(
        self,
        x: torch.Tensor,
        masking_type: str = "random",
        mask_ratio: float = 0.75,
        ground_truth: Optional[torch.Tensor] = None,
    ) -> Any:
        x = self.patch_embed(x)

        x = x + self.pos_embed[:, 1:, :]

        if masking_type == "random":
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        elif masking_type == "ground_truth":
            assert ground_truth is not None
            ground_truth = self.patchify(ground_truth)
            ground_truth = (ground_truth.sum(dim=-1) > 0).float()
            x, mask, ids_restore = self.mask_specific(x, ground_truth, mask_ratio)
        else:
            x, mask, ids_restore = self.block_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(
        self, x: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)

        x = x[:, 1:, :]

        return x

    def forward_loss(
        self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        imgs: [N, 3, D, H, W]
        pred: [N, L, p*p*p *3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(
        self,
        x: torch.Tensor,
        masking_type: str = "random",
        ground_truth: Optional[torch.Tensor] = None,
        mask_ratio: float = 0.75,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, mask, ids_restore = self.forward_encoder(
            x,
            masking_type=masking_type,
            mask_ratio=mask_ratio,
            ground_truth=ground_truth,
        )
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(x, pred, mask)

        return loss, pred, mask


def brain2vec_base_patch8_dec512d8b(**kwargs: Any) -> nn.Module:
    return Brain2Vec(
        patch_size=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def brain2vec_large_patch8_dec512d8b(**kwargs: Any) -> nn.Module:
    return Brain2Vec(
        patch_size=8,
        embed_dim=1056,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def brain2vec_huge_patch8_dec512d8b(**kwargs: Any) -> nn.Module:
    return Brain2Vec(
        patch_size=8,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
