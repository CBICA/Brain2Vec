from functools import partial
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Brain2Vec.models.brain2vec import Brain2Vec


def dice_coeff(
    input: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    assert input.shape == target.shape, (
        f"Dice input/target shape mismatch: " f"{input.shape} vs {target.shape}"
    )

    intersection = (input * target).sum(dim=(2, 3, 4))

    input_sum = input.sum(dim=(2, 3, 4))
    target_sum = target.sum(dim=(2, 3, 4))

    dice = (2.0 * intersection + epsilon) / (input_sum + target_sum + epsilon)

    return dice.mean()


def multiclass_dice_coeff(
    input: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:

    return dice_coeff(
        input,
        target,
        epsilon=epsilon,
    )


def dice_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    multiclass: bool = False,
) -> torch.Tensor:
    if multiclass:
        return 1.0 - multiclass_dice_coeff(input, target)

    return 1.0 - dice_coeff(input, target)


class ViTSegmenter(Brain2Vec):
    def __init__(self, n_classes: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.n_classes = n_classes
        decoder_embed_dim: int = kwargs.get("decoder_embed_dim", 512)
        patch_size: int = kwargs.get("patch_size", 8)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**3 * n_classes, bias=True
        )

    def unpatchify_logits(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        ph, pw, pd = self.patch_embed.patch_size
        C = self.n_classes

        g = round(L ** (1 / 3))
        assert g**3 == L, f"Number of patches {L} is not a perfect cube"

        x = x.reshape(B, g, g, g, ph, pw, pd, C)
        x = torch.einsum("bdhwpqrc->bcdphqwr", x)

        return x.reshape(B, C, g * ph, g * pw, g * pd)

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
    ) -> torch.Tensor:
        x = self.decoder_embed(x) + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_pred(x)[:, 1:, :]

        return x

    def forward_segmenter_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        target = target.squeeze(1)
        
        ce_loss = F.cross_entropy(pred, target)

        _dice_loss = dice_loss(
            F.softmax(pred, dim=1).float(),
            F.one_hot(target, self.n_classes).squeeze(0).permute(0, 4, 1, 2, 3).float(),
            multiclass=True,
        )

        return ce_loss + _dice_loss

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        latent, _, _ = self.forward_encoder(x)

        pred = self.unpatchify_logits(self.forward_decoder(latent))
        loss = self.forward_segmenter_loss(pred, target)

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
