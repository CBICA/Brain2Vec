# Copyright 2026 CBICA
#
# Author: Spiros Maggioros
#
# Licenced under CBICA's non-Commercial Software Agreement:
# https://www.med.upenn.edu/cbica/software-agreement-non-commercial.html

import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset


def get_prefix(img_name: str) -> str:
    """
    Returns the prefix(the name or label) of the images. Note that, the images
    have a suffix of .nii.gz and the dlicv and LPS orientation labels.
    """
    return str(img_name[: len(img_name) - 28])


def create_loader(
    loader: Optional[Dataset],
    batch_size: int,
    shuffle: bool,
    distributed: bool = False,
    num_tasks: Optional[int] = None,
    global_rank: Optional[int] = None,
) -> Optional[DataLoader]:
    if loader is None:
        return None

    if distributed:
        assert num_tasks is not None
        assert global_rank is not None
        sampler = torch.utils.data.DistributedSampler(
            loader,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=shuffle,
        )
        dataloader = DataLoader(
            loader,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=os.cpu_count() // 4,  # type: ignore
            prefetch_factor=2,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
        )
    else:
        dataloader = DataLoader(
            loader,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=os.cpu_count() // 4,  # type: ignore
            collate_fn=lambda x: torch.utils.data.dataloader.default_collate(x),
            prefetch_factor=2,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
        )
    return dataloader
