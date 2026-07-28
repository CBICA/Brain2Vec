# Copyright 2026 CBICA
#
# Author: Spiros Maggioros
#
# Licenced under CBICA's non-Commercial Software Agreement:
# https://www.med.upenn.edu/cbica/software-agreement-non-commercial.html

import os
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torchio as tio
from torch.utils.data import DataLoader, Dataset
from typing_extensions import Literal


class NiftiLoader(Dataset):
    """
    Class for the SingleSubject Dataloader. Returns the input tensor with it's label(as a torch.tensor)

    :param mode: train(for ViT fine-tuning), pretrain(for Brain2Vec) or inference
    :type mode: str
    :param in_dir: The input directory that contains train/ test/ and eval/ folders
    :type in_dir: str
    :param label_dict: A dictionary that maps MRID to target value
    :type label_dict: Optional[dict]
    :param data_augmentation: True if you want data augmentation to be applied, False otherwise
    :type data_augmentation: bool
    """

    def __init__(
        self,
        mode: Literal["train", "pretrain", "inference"],
        in_dir: str,
        label_dict: Optional[dict] = None,
        data_augmentation: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        assert mode in ["train", "pretrain", "inference"]
        if label_dict is not None:
            assert (
                mode != "pretrain"
            ), "Can't perform inference with a passed label dict."
        else:
            assert (
                mode == "pretrain" or mode == "inference"
            ), "Can't perform training or evaluation without a label dict."

        self.in_dir = Path(in_dir)
        self.data_augmentation = data_augmentation
        self.mode = mode
        self.label_dict = label_dict
        self.images: list = []
        self.img_label: list = []
        self.mrids: list = []
        self.img_size = kwargs.get("img_size", (128, 128, 128))
        self.transforms = [
            tio.RandomBlur(2),
            tio.RandomGamma(0.3),
            tio.RandomFlip(axes=("LR",)),
            tio.RandomNoise(mean=0, std=2),
            tio.Affine(
                scales=(1, 1, 1),
                degrees=tuple(np.random.uniform(low=-40, high=40, size=3)),
                translation=tuple(np.random.uniform(low=-10, high=10, size=3)),
                image_interpolation="linear",
                default_pad_value="minimum",
            ),
        ]
        self.resize = tio.Resize(self.img_size)
        self._fetch_images()
        if mode == "train":
            self._fetch_labels()

    def _fetch_images(self) -> None:
        for img_name in os.listdir(self.in_dir):
            if img_name.endswith(".nii.gz"):
                image_path = self.in_dir / img_name
                self.images.append(image_path)
                self.mrids.append(img_name)

    def _fetch_labels(self) -> None:
        for img in self.images:
            if self.label_dict is not None:
                label = self.label_dict.get(img.name, 0)
            self.img_label.append((img, label))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Any:
        if self.mode == "train":
            img_path, label = self.img_label[index]
        else:
            img_path = self.images[index]
            mrid = self.mrids[index]

        img = tio.ScalarImage(img_path)
        img = self.resize(img)

        if self.data_augmentation:
            sample = [random.randint(0, 4) for _ in range(len(self.transforms))]
            selected_transformations = [
                self.transforms[i] for (i, _is) in enumerate(sample) if _is == 1
            ]
            transforms_total = tio.Compose(selected_transformations)
            if len(selected_transformations) > 0:
                img = transforms_total(img)

        img_tensor = torch.tensor(img.data, dtype=torch.float32)

        if self.mode == "train":
            return img_tensor, torch.tensor(label, dtype=torch.float32)
        else:
            return img_tensor, mrid


def create_dataloaders(
    mode: str,
    in_dir: str,
    batch_size: int,
    distributed: bool = False,
    target: Optional[str] = None,
    label_dict_csv: Optional[str] = None,
    num_tasks: Optional[int] = None,
    global_rank: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """
    Creates the train, test and eval dataloader

    :param mode: Either train, pretrain or inference
    :type model: str
    :param in_dir: The input directory that contains train/ test/ and eval/ folders
    :type in_dir: str
    :param batch_size: The passed batch_size that the data will be splitted into
    :type batch_size: int
    :param label_dict_csv: The complete path to the .csv that contains the MRID and the target columns
    :type label_dict: str
    """

    train_dir = f"{in_dir}/train"
    test_dir = f"{in_dir}/test"
    eval_dir = f"{in_dir}/eval"
    img_size = kwargs.get("img_size", (128, 128, 128))

    if mode == "finetune":
        assert label_dict_csv is not None
        assert target is not None
        in_csv = pd.read_csv(label_dict_csv)
        label_dict = {row["MRID"]: row[target] for _, row in in_csv.iterrows()}

    if os.path.isdir(train_dir):
        train_loader = NiftiLoader(
            mode="pretrain" if mode == "pretrain" else "train",
            in_dir=train_dir,
            label_dict=None if mode == "pretrain" else label_dict,
            data_augmentation=True,
            img_size=img_size,
        )
    else:
        train_loader = None

    if os.path.isdir(test_dir):
        test_loader = NiftiLoader(
            mode="pretrain" if mode == "pretrain" else "train",
            in_dir=test_dir,
            label_dict=None if mode == "pretrain" else label_dict,
            data_augmentation=False,
            img_size=img_size,
        )
    else:
        test_loader = None

    if os.path.isdir(eval_dir):
        eval_loader = NiftiLoader(
            mode="inference",
            in_dir=eval_dir,
            label_dict=None if mode == "pretrain" else label_dict,
            data_augmentation=False,
            img_size=img_size,
        )
    else:
        eval_loader = None

    if train_loader is not None:
        if distributed:
            assert num_tasks is not None
            assert global_rank is not None
            train_sampler = torch.utils.data.DistributedSampler(
                train_loader,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=True,
            )
            train_dataloader = DataLoader(
                train_loader,
                sampler=train_sampler,
                batch_size=batch_size,
                num_workers=os.cpu_count() // 4,  # type: ignore
                prefetch_factor=2,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=True,
            )
        else:
            train_dataloader = DataLoader(
                train_loader,
                batch_size=batch_size,
                shuffle=True,
                num_workers=os.cpu_count() // 4,  # type: ignore
                collate_fn=lambda x: torch.utils.data.dataloader.default_collate(x),
                prefetch_factor=2,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=True,
            )
    else:
        train_dataloader = None

    if test_loader is not None:
        if distributed:
            assert num_tasks is not None
            assert global_rank is not None
            test_sampler = torch.utils.data.DistributedSampler(
                test_loader,
                num_replicas=num_tasks,
                rank=global_rank,
            )
            test_dataloader = DataLoader(
                test_loader,
                sampler=test_sampler,
                batch_size=batch_size,
                num_workers=os.cpu_count() // 4,  # type: ignore
                prefetch_factor=2,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=True,
            )
        else:
            test_dataloader = DataLoader(
                test_loader,
                batch_size=batch_size,
                num_workers=os.cpu_count() // 4,  # type: ignore
                collate_fn=lambda x: torch.utils.data.dataloader.default_collate(x),
                prefetch_factor=2,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=True,
            )
    else:
        test_dataloader = None

    if eval_loader is not None:
        if distributed:
            assert num_tasks is not None
            assert global_rank is not None
            eval_sampler = torch.utils.data.DistributedSampler(
                eval_loader,
                num_replicas=num_tasks,
                rank=global_rank,
            )
            eval_dataloader = DataLoader(
                eval_loader,
                sampler=eval_sampler,
                batch_size=1,
                num_workers=os.cpu_count() // 4,  # type: ignore
                prefetch_factor=2,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=True,
            )
        else:
            eval_dataloader = DataLoader(
                eval_loader,
                batch_size=1,
                num_workers=os.cpu_count() // 4,  # type: ignore
                collate_fn=lambda x: torch.utils.data.dataloader.default_collate(x),
                prefetch_factor=2,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=True,
            )
    else:
        eval_dataloader = None

    return train_dataloader, test_dataloader, eval_dataloader
