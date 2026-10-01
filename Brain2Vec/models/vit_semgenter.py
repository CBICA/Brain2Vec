from functools import partial
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Brain2Vec.models.brain2vec import Brain2Vec


def dice_coeff(
    input: torch.Tensor,
    target: torch.Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(
    input: torch.Tensor,
    target: torch.Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    return dice_coeff(
        input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon
    )


def dice_loss(
    input: torch.Tensor, target: torch.Tensor, multiclass: bool = False
) -> torch.Tensor:
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


class ViTSegmenter(Brain2Vec):
    def __init__(self, n_classes: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.n_classes = n_classes

    def forward_encoder(
        self,
        x: torch.Tensor,
        masking_type: str = "random",
        mask_ratio: float = 0.0,
        ground_truth: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.patch_embed(x)

        x = x + self.pos_embed[:, 1:, :]

        x, mask, ids_restore = self.random_masking(x, mask_ratio=0.0)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
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

    def forward_segmenter_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        assert pred.shape == target.shape, "Prediction shape != Target shape"
        target = self.patchify(target)

        print(f"One hot shape: {(F.one_hot(target, self.n_classes)).shape}")

        ce_loss = F.cross_entropy(pred, target)
        _dice_loss = dice_loss(
            F.softmax(pred, dim=1).float(),
            F.one_hot(target, self.n_classes).float(),
            multiclass=True,
        )

        return ce_loss + _dice_loss

    def forward(
        self,
        x: torch.Tensor,
        masking_type: str = "random",
        ground_truth: Optional[torch.Tensor] = None,
        mask_ratio: float = 0.0,
    ) -> torch.Tensor:

        assert masking_type == "random"
        assert ground_truth is None
        assert mask_ratio == 0.0

        latent, _, ids_restore = self.forward_encoder(x)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_segmenter_loss(pred, x)

        return loss


def vit_segmenter_base_patch8_dec512d8b(**kwargs: Any) -> nn.Module:
    return ViTSegmenter(
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


def vit_segmenter_large_patch8_dec512d8b(**kwargs: Any) -> nn.Module:
    return ViTSegmenter(
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


def vit_segmenter_huge_patch8_dec512d8b(**kwargs: Any) -> nn.Module:
    return ViTSegmenter(
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
