# Copyright 2026 CBICA
#
# Author: Spiros Maggioros
#
# Licenced under CBICA's non-Commercial Software Agreement:
# https://www.med.upenn.edu/cbica/software-agreement-non-commercial.html

import argparse
from typing import Any, Tuple

import numpy as np
import torch
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy
from torch import nn
from torch.distributed import destroy_process_group

from Brain2Vec.dataloader.loader import create_dataloaders
from Brain2Vec.models.brain2vec import (
    brain2vec_base_patch8_dec512d8b,
    brain2vec_huge_patch8_dec512d8b,
    brain2vec_large_patch8_dec512d8b,
)
from Brain2Vec.models.utils import freeze_early_layers, load_brain2vec
from Brain2Vec.models.vit import vit_base_patch8, vit_huge_patch8, vit_large_patch8
from Brain2Vec.models.vit_segmenter import (
    vit_segmenter_base_patch8_dec512d8b,
    vit_segmenter_huge_patch8_dec512d8b,
    vit_segmenter_large_patch8_dec512d8b,
)
from Brain2Vec.trainer.brain2vec_pretrain import train as brain2vec_pretrain
from Brain2Vec.trainer.segmenter_trainer import train as segmenter_trainer
from Brain2Vec.trainer.utils import init_distributed_mode
from Brain2Vec.trainer.vit_finetune import train as vit_finetune


class MixUp:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if self.alpha > 0.0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda()

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam


class mixup_criterion:
    def __init__(self, criterion: nn.Module) -> None:
        self.criterion = criterion

    def __call__(
        self, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float
    ) -> torch.Tensor:
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)


def available_models() -> dict:
    return {
        "brain2vec_base": brain2vec_base_patch8_dec512d8b,
        "brain2vec_large": brain2vec_large_patch8_dec512d8b,
        "brain2vec_huge": brain2vec_huge_patch8_dec512d8b,
        "vit_base": vit_base_patch8,
        "vit_large": vit_large_patch8,
        "vit_huge": vit_huge_patch8,
        "segmenter_base": vit_segmenter_base_patch8_dec512d8b,
        "segmenter_large": vit_segmenter_large_patch8_dec512d8b,
        "segmenter_huge": vit_segmenter_huge_patch8_dec512d8b,
    }


def select_optimizer(
    model: nn.Module, optimizer: str, lr: float, weight_decay: float
) -> torch.optim:
    AVAILABLE_OPTIMIZERS = {
        "Adam": torch.optim.Adam(
            params=model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        ),
        "AdamW": torch.optim.AdamW(
            params=model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        ),
        "Adadelta": torch.optim.Adadelta(
            params=model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        ),
        "SGD": torch.optim.SGD(
            params=model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        ),
    }  # We support a limited amount of optimizers as they are the ones that we tested on our task and had the best results
    assert (
        optimizer in AVAILABLE_OPTIMIZERS.keys()
    ), f"Please provide a currently supported optimizer: {AVAILABLE_OPTIMIZERS.keys()}"
    return AVAILABLE_OPTIMIZERS[optimizer]


def run_training_assertions(args: Any) -> None:
    assert args.lr > 0.0
    assert args.weight_decay >= 0.0
    assert args.batch_size > 0
    assert args.epochs > 0
    assert args.patience >= 0 and args.patience < args.epochs
    assert args.accum_iter >= 1


def run_brain2vec_pretrain(args: Any) -> None:
    run_training_assertions(args)
    init_distributed_mode(args)

    avail_models = available_models()
    assert args.model in avail_models.keys()

    model = avail_models[args.model](img_size=args.img_size).to(args.device)
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = model.to(args.device)

    model_without_ddp = model

    num_tasks = args.world_size
    global_rank = args.rank
    train_dataloader, test_dataloader, _ = create_dataloaders(
        mode="pretrain",
        in_dir=args.in_dir,
        batch_size=args.batch_size,
        loader_type="default",
        target=None,
        distributed=args.distributed,
        num_tasks=num_tasks if args.distributed else None,
        global_rank=global_rank if args.distributed else None,
        img_size=(int(args.img_size),) * 3,
    )

    eff_batch_size = args.batch_size * args.world_size
    eff_lr = args.lr * eff_batch_size / 32

    print("Base lr: %.2e" % args.lr, force=True, flush=True)  # type: ignore
    print("Actual lr: %.2e" % eff_lr, force=True, flush=True)  # type: ignore
    print(f"effective batch size: {eff_batch_size}", force=True, flush=True)  # type: ignore

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=True,
        )
        model_without_ddp = model.module

    optimizer = select_optimizer(
        model=model,
        optimizer=args.optimizer,
        lr=eff_lr,
        weight_decay=args.weight_decay,
    )

    _ = brain2vec_pretrain(
        model=model,
        model_without_ddp=model_without_ddp,
        distributed=args.distributed,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        masking_type=args.masking_type,
        mask_ratio=args.mask_ratio,
        optimizer=optimizer,
        epochs=args.epochs,
        patience=args.patience,
        accum_iter=args.accum_iter,
        target_dir=args.out_dir,
        model_name=args.model_name,
        device=args.gpu,
        verbose=args.verbose,
    )


def run_vit_finetune(args: Any) -> None:
    run_training_assertions(args)
    init_distributed_mode(args)

    avail_models = available_models()
    assert args.model in avail_models.keys()

    model_class = avail_models[args.model]
    if len(args.pretrain) > 0:
        model = load_brain2vec(
            repo_id="cbica/Brain2Vec",
            model_class=model_class,
            model_kwargs={
                "img_size": args.img_size,
                "num_classes": 1 if args.num_classes == -1 else args.num_classes,
            },
            weights_filename=args.pretrain,
            device=args.device,
            strict=False,
        )
    else:
        model = model_class(
            img_size=args.img_size,
            num_classes=1 if args.num_classes == -1 else args.num_classes,
        )
    if args.freeze_ratio > 0.0:
        freeze_early_layers(model, args.freeze_ratio)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = model.to(args.device)

    model_without_ddp = model
    num_tasks = args.world_size
    global_rank = args.rank

    train_dataloader, test_dataloader, _ = create_dataloaders(
        mode="finetune",
        in_dir=args.in_dir,
        batch_size=args.batch_size,
        loader_type="default",
        label_dict_csv=args.label_dict,
        target=args.target,
        distributed=args.distributed,
        num_tasks=num_tasks if args.distributed else None,
        global_rank=global_rank if args.distributed else None,
        img_size=(int(args.img_size),) * 3,
    )

    eff_batch_size = args.batch_size * args.world_size * args.accum_iter
    eff_lr = args.lr * eff_batch_size / 32

    print("Base lr: %.2e" % args.lr, force=True, flush=True)  # type: ignore
    print("Actual lr: %.2e" % eff_lr, force=True, flush=True)  # type: ignore
    print(f"Accumulate grad iterations: {args.accum_iter}", force=True, flush=True)  # type: ignore
    print(f"effective batch size: {eff_batch_size}", force=True, flush=True)  # type: ignore

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=True,
        )
        model_without_ddp = model.module

    mixup = None
    if args.mixup > 0.0:
        print("Mixup Activated!")
        mixup = MixUp(alpha=args.mixup)

    optimizer = select_optimizer(
        model=model,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if mixup is not None:
        if args.mode == "binary":
            loss_fn = mixup_criterion(criterion=torch.nn.BCEWithLogitsLoss())
        elif args.mode == "multiclass":
            loss_fn = mixup_criterion(criterion=torch.nn.CrossEntropyLoss())
        else:
            loss_fn = mixup_criterion(criterion=torch.nn.L1Loss(reduction="mean"))
    elif args.smoothing > 0.0:
        assert (
            args.mode != "regression"
        ), "Can't perform label smoothing with regression labels"
        loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        if args.mode == "binary":
            loss_fn = torch.nn.BCEWithLogitsLoss()
        elif args.mode == "multiclass":
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            loss_fn = torch.nn.L1Loss(reduction="mean")
    _ = vit_finetune(
        model=model,
        mode=args.mode,
        model_without_ddp=model_without_ddp,
        distributed=args.distributed,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        mixup_fn=mixup,
        loss_fn=loss_fn,
        epochs=args.epochs,
        patience=args.patience,
        num_classes=None if args.num_classes == -1 else args.num_classes,
        target_dir=args.out_dir,
        model_name=args.model_name,
        device=args.device,
        verbose=args.verbose,
    )


def run_segmenter_trainer(args: Any) -> None:
    run_training_assertions(args)
    init_distributed_mode(args)

    avail_models = available_models()
    assert args.model in avail_models.keys()

    model = avail_models[args.model](
        img_size=args.img_size, n_classes=args.n_classes
    ).to(args.device)
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = model.to(args.device)

    model_without_ddp = model

    num_tasks = args.world_size
    global_rank = args.rank
    train_dataloader, test_dataloader, _ = create_dataloaders(
        mode="pretrain",
        in_dir=args.in_dir,
        batch_size=args.batch_size,
        loader_type="segmenter",
        target=None,
        distributed=args.distributed,
        num_tasks=num_tasks if args.distributed else None,
        global_rank=global_rank if args.distributed else None,
        img_size=(int(args.ims_size),) * 3,
    )

    eff_batch_size = args.batch_size * args.world_size
    eff_lr = args.lr * eff_batch_size / 32

    print("Base lr: %.2e" % args.lr, force=True, flush=True)  # type: ignore
    print("Actual lr: %.2e" % eff_lr, force=True, flush=True)  # type: ignore
    print(f"effective batch size: {eff_batch_size}", force=True, flush=True)  # type: ignore

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=True,
        )
        model_without_ddp = model.module

    optimizer = select_optimizer(
        model=model,
        optimizer=args.optimizer,
        lr=eff_lr,
        weight_decay=args.weight_decay,
    )

    _ = segmenter_trainer(
        model=model,
        model_without_ddp=model_without_ddp,
        distributed=args.distributed,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        epochs=args.epochs,
        patience=args.patience,
        accum_iter=args.accum_iter,
        device=args.device,
        target_dir=args.out_dir,
        model_name=args.model_name,
        verbose=args.verbose,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="Brain2Vec",
        add_help=False,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    brain2vec_train = subparsers.add_parser(
        "brain2vec_pretrain", help="Select Brain2Vec pre-training mode"
    )
    brain2vec_train.add_argument(
        "--img_size",
        type=int,
        default=128,
        required=False,
    )

    brain2vec_train.add_argument(
        "--masking_type",
        type=str,
        default="random",
        required=False,
    )

    brain2vec_train.add_argument(
        "--mask_ratio",
        type=float,
        default=0.75,
        required=False,
    )

    brain2vec_train.add_argument(
        "--model",
        type=str,
        default="brain2vec_base",
        required=False,
    )

    brain2vec_train.add_argument(
        "--in_dir",
        type=str,
        required=True,
    )

    brain2vec_train.add_argument(
        "--batch_size",
        type=int,
        default=32,
        required=False,
    )

    brain2vec_train.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default=".",
        required=False,
    )

    brain2vec_train.add_argument(
        "--model_name",
        type=str,
        default="brain2vec_best.pth",
        required=False,
    )

    brain2vec_train.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda",
        required=False,
    )

    brain2vec_train.add_argument(
        "--lr",
        type=float,
        default=0.01,
        required=False,
    )

    brain2vec_train.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        required=False,
    )

    brain2vec_train.add_argument(
        "--epochs",
        type=int,
        default=200,
        required=False,
    )

    brain2vec_train.add_argument(
        "--patience",
        type=int,
        default=10,
        required=False,
    )

    brain2vec_train.add_argument(
        "--optimizer",
        type=str,
        default="Adadelta",
        required=False,
    )

    brain2vec_train.add_argument(
        "--accum_iter",
        type=int,
        default=1,
        required=False,
    )

    brain2vec_train.add_argument(
        "--verbose",
        action="store_true",
    )

    brain2vec_train.add_argument("--distributed", action="store_true")

    brain2vec_train.add_argument(
        "--world_size",
        type=int,
        default=1,
        required=False,
    )

    brain2vec_train.add_argument(
        "--dist_url",
        default="env://",
    )
    brain2vec_train.set_defaults(func=run_brain2vec_pretrain)

    vit_finetune = subparsers.add_parser(
        "vit_finetune", help="Select ViT fine-tuning mode"
    )
    vit_finetune.add_argument(
        "--img_size",
        type=int,
        default=128,
        required=False,
    )

    vit_finetune.add_argument(
        "--mode",
        type=str,
        required=True,
        help="Either regression, binary or multiclass",
    )

    vit_finetune.add_argument(
        "--model",
        type=str,
        default="vit_base",
        required=False,
    )

    vit_finetune.add_argument(
        "--num_classes",
        type=int,
        default=-1,
        required=False,
    )

    vit_finetune.add_argument(
        "--target",
        type=str,
        required=False,
    )

    vit_finetune.add_argument(
        "--pretrain",
        type=str,
        default="",
        required=False,
    )

    vit_finetune.add_argument(
        "--in_dir",
        type=str,
        required=True,
    )

    vit_finetune.add_argument(
        "--label_dict",
        type=str,
        required=False,
    )

    vit_finetune.add_argument(
        "--batch_size",
        type=int,
        default=32,
        required=False,
    )

    vit_finetune.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default=".",
        required=False,
    )

    vit_finetune.add_argument(
        "--model_name",
        type=str,
        default="vit_finetune_best.pth",
    )

    vit_finetune.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda",
        required=False,
    )

    vit_finetune.add_argument(
        "--lr",
        type=float,
        default=0.01,
        required=False,
    )

    vit_finetune.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        required=False,
    )

    vit_finetune.add_argument(
        "--epochs",
        type=int,
        default=200,
        required=False,
    )

    vit_finetune.add_argument(
        "--patience",
        type=int,
        default=10,
        required=False,
    )

    vit_finetune.add_argument(
        "--optimizer",
        type=str,
        default="Adadelta",
        required=False,
    )

    vit_finetune.add_argument(
        "--accum_iter",
        type=int,
        default=1,
        required=False,
    )

    vit_finetune.add_argument(
        "--freeze_ratio",
        type=float,
        default=0.0,
    )

    # augmentation specifics
    vit_finetune.add_argument(
        "--smoothing",
        type=float,
        default=0.0,
    )

    # mixup specifics
    vit_finetune.add_argument(
        "--mixup",
        type=float,
        default=0,
    )

    # misc
    vit_finetune.add_argument(
        "--verbose",
        action="store_true",
    )

    vit_finetune.add_argument(
        "--dist_url",
        default="env://",
    )

    vit_finetune.add_argument("--distributed", action="store_true")
    vit_finetune.set_defaults(func=run_vit_finetune)

    segmenter_trainer = subparsers.add_parser(
        "segmenter_trainer", help="Select Segmenter trainer mode"
    )
    segmenter_trainer.add_argument(
        "--img_size",
        type=int,
        default=128,
        required=False,
    )

    segmenter_trainer.add_argument(
        "--model",
        type=str,
        default="segmenter_base",
        required=False,
    )

    segmenter_trainer.add_argument(
        "--n_classes",
        type=int,
        required=True,
    )

    segmenter_trainer.add_argument(
        "--in_dir",
        type=str,
        required=True,
    )

    segmenter_trainer.add_argument(
        "--batch_size",
        type=int,
        default=32,
        required=False,
    )

    segmenter_trainer.add_argument(
        "--o",
        "--out_dir",
        type=str,
        default=".",
        required=False,
    )

    segmenter_trainer.add_argument(
        "--model_name",
        type=str,
        default="vit_segmenter_best.pth",
    )

    segmenter_trainer.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda",
        required=False,
    )

    segmenter_trainer.add_argument(
        "--lr",
        type=float,
        default=0.01,
        required=False,
    )

    segmenter_trainer.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        required=False,
    )

    segmenter_trainer.add_argument(
        "--epochs",
        type=int,
        default=200,
        required=False,
    )

    segmenter_trainer.add_argument(
        "--patience",
        type=int,
        default=10,
        required=False,
    )

    segmenter_trainer.add_argument(
        "--optimizer",
        type=str,
        default="Adadelta",
        required=False,
    )

    segmenter_trainer.add_argument(
        "--accum_iter",
        type=int,
        default=1,
        required=False,
    )

    segmenter_trainer.add_argument(
        "--verbose",
        action="store_true",
    )

    segmenter_trainer.add_argument(
        "--dist_url",
        default="env://",
    )

    segmenter_trainer.add_argument("--distributed", action="store_true")
    segmenter_trainer.set_defaults(func=run_segmenter_trainer)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
    destroy_process_group()
