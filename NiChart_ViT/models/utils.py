import numpy as np
import torch
import torch.nn as nn


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 8,
        num_hidden: int = 512,
    ) -> None:
        super().__init__()

        self.img_size, self.patch_size = (img_size, img_size, img_size), (
            patch_size,
            patch_size,
            patch_size,
        )
        self.num_patches = (img_size // patch_size) ** 3
        self.conv = nn.Conv3d(1, num_hidden, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).flatten(2).transpose(1, 2)


def get_3d_sincos_pos_embed(
    embed_dim: int, grid_size: int, cls_token: bool = False
) -> np.ndarray:
    """
    grid_size: int of the grid depth, height and width
    return:
    pos_embed: [grid_size*grid_size*grid_size, embed_dim] or [1+grid_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_d = np.arange(grid_size, dtype=np.float32)
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h, grid_d)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size, grid_size, grid_size])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    # Distribute dimensions as evenly as possible across 3 axes
    # For embed_dim=768: 256, 256, 256
    embed_dim_d = embed_dim // 3
    embed_dim_h = embed_dim // 3
    embed_dim_w = embed_dim - embed_dim_d - embed_dim_h

    # use distributed dimensions to encode each axis
    emb_d = get_1d_sincos_pos_embed_from_grid(
        embed_dim_d, grid[2]
    )  # (D*H*W, embed_dim_d)
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim_h, grid[1]
    )  # (D*H*W, embed_dim_h)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim_w, grid[0]
    )  # (D*H*W, embed_dim_w)

    emb = np.concatenate([emb_d, emb_h, emb_w], axis=1)  # (D*H*W, embed_dim)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> np.ndarray:
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
