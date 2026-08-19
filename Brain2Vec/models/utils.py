# Copyright 2026 CBICA
#
# Author: Spiros Maggioros
#
# References:
#   facebookresearch: https://github.com/facebookresearch/mae/tree/main/util
#
# Licenced under CBICA's non-Commercial Software Agreement:
# https://www.med.upenn.edu/cbica/software-agreement-non-commercial.html

import json
from pathlib import Path
from typing import Any, Optional, Type

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


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


def download_file(
    repo_id: str,
    filename: str,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> Path:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
    )
    return Path(path)


def load_config(
    repo_id: str,
    filename: str = "config.json",
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> Any:
    try:
        path = download_file(repo_id, filename, revision, cache_dir, token)
    except Exception:
        return {}
    with open(path) as f:
        return json.load(f)


def load_brain2vec(
    repo_id: str,
    model_class: Type[torch.nn.Module],
    model_kwargs: Optional[dict[str, Any]] = None,
    weights_filename: str = "brain2vec.safetensors",
    config_filename: str = "config.json",
    revision: Optional[str] = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> torch.nn.Module:
    kwargs = (
        model_kwargs
        if model_kwargs is not None
        else load_config(repo_id, config_filename, revision)
    )

    model = model_class(**kwargs)

    weights_path = download_file(repo_id, weights_filename, revision)
    state_dict = load_file(str(weights_path), device=str(device))

    if strict:
        model.load_state_dict(state_dict)
    else:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[load_brain2vec] missing keys: {missing}")
        if unexpected:
            print(f"[load_brain2vec] unexpected keys: {unexpected}")

    model.to(device)
    return model
