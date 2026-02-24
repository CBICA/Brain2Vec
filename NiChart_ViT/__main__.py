import argparse
from typing import Any, Optional, Tuple

import numpy as np
import torch
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy
from torch import nn
from torch.distributed import destroy_process_group

from NiChart_ViT.dataloader.loader import create_dataloaders
from NiChart_ViT.models.mae import (
    mae_vit_base_patch8_dec512d8b,
    mae_vit_huge_patch8_dec512d8b,
    mae_vit_large_patch8_dec512d8b,
)
from NiChart_ViT.models.vit import (
    barlow_twins_base,
    barlow_twins_huge,
    barlow_twins_large,
    vit_base_patch8,
    vit_huge_patch8,
    vit_large_patch8,
)
from NiChart_ViT.trainer.mae_pretrain import train as mae_pretrain
from NiChart_ViT.trainer.utils import init_distributed_mode, load_weights
from NiChart_ViT.trainer.vit_finetune import train as vit_finetune
from NiChart_ViT.trainer.vit_ssl import train as train_vit_bt


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


def available_models(
    img_size: tuple = (128, 128, 128),
    patch_size: tuple = (8, 8, 8),
    p_slices: float = 1.0,
    num_classes: Optional[int] = None,
    projector: Optional[str] = None,
    eff_batch_size: Optional[int] = None,
    _lambda: Optional[float] = None,
    device: str = "cuda",
) -> dict:

    if p_slices < 1.0:
        img_size = (int(img_size[0] * p_slices), img_size[1], img_size[2])
    return {
        "mae_base": mae_vit_base_patch8_dec512d8b(img_size=img_size).to(device),
        "mae_large": mae_vit_large_patch8_dec512d8b(img_size=img_size).to(device),
        "mae_huge": mae_vit_huge_patch8_dec512d8b(img_size=img_size).to(device),
        "vit_base": (
            vit_base_patch8(img_size=img_size, patch_size=patch_size)
            if num_classes is None
            else vit_base_patch8(img_size=img_size, num_classes=num_classes).to(device)
        ),
        "vit_large": (
            vit_large_patch8(img_size=img_size, patch_size=patch_size,)
            if num_classes is None
            else vit_large_patch8(img_size=img_size, num_classes=num_classes).to(device)
        ),
        "vit_huge": (
            vit_huge_patch8(img_size=img_size, patch_size=patch_size,)
            if num_classes is None
            else vit_huge_patch8(img_size=img_size, num_classes=num_classes).to(device)
        ),
        "bt_base": barlow_twins_base(
            img_size=img_size,
            patch_size=patch_size,
            projector=projector,
            eff_batch_size=eff_batch_size,
            _lambda=_lambda,
        ),
        "bt_large": barlow_twins_large(
            img_size=img_size,
            patch_size=patch_size,
            projector=projector,
            eff_batch_size=eff_batch_size,
            _lambda=_lambda,
        ),
        "bt_huge": barlow_twins_huge(
            img_size=img_size,
            patch_size=patch_size,
            projector=projector,
            eff_batch_size=eff_batch_size,
            _lambda=_lambda,
        ),
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


def run_mae_pretrain(args: Any) -> None:
    assert args.lr > 0.0
    assert args.weight_decay >= 0.0
    assert args.batch_size > 0
    assert args.epochs > 0
    assert args.patience >= 0 and args.patience < args.epochs
    assert args.accum_iter >= 1

    avail_models = available_models(img_size=args.img_size, device=args.device)
    assert args.model in avail_models.keys()

    init_distributed_mode(args)

    model = avail_models[args.model]
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
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    optimizer = select_optimizer(
        model=model,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    _ = mae_pretrain(
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
    assert args.lr > 0.0
    assert args.weight_decay >= 0.0
    assert args.batch_size > 0
    assert args.epochs > 0
    assert args.patience >= 0 and args.patience < args.epochs
    assert args.accum_iter >= 1

    init_distributed_mode(args)

    img_size = args.img_size.split("-")
    assert len(img_size) == 3
    img_size = (int(img_size[0]), int(img_size[1]), int(img_size[2]))

    patch_size = args.patch_size.split("-")
    assert len(patch_size) == 3
    patch_size = (int(patch_size[0]), int(patch_size[1]), int(patch_size[2]))

    avail_models = available_models(
        img_size=img_size,
        patch_size=patch_size,
        p_slices=args.p_slices,
        num_classes=1 if args.num_classes == -1 else args.num_classes,
        projector=args.projector if args.mode == "ssl" else None,
        eff_batch_size=args.batch_size * args.world_size * args.accum_iter,
        _lambda=args.lambd,
        device=args.device,
    )
    assert args.model in avail_models.keys()

    model = avail_models[args.model]
    if len(args.pretrain) > 0:
        load_weights(model=model, model_weights=args.pretrain, device=args.device)
    # otherwise just train from scratch

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = model.to(args.device)

    model_without_ddp = model
    num_tasks = args.world_size
    global_rank = args.rank

    train_dataloader, test_dataloader, _ = create_dataloaders(
        mode="ssl" if args.mode == "ssl" else "finetune",
        in_dir=args.in_dir,
        batch_size=args.batch_size,
        label_dict_csv=args.label_dict if args.mode == "finetune" else None,
        target=args.target if args.mode == "finetune" else None,
        distributed=args.distributed,
        num_tasks=num_tasks if args.distributed else None,
        global_rank=global_rank if args.distributed else None,
        img_size=img_size,
        p_slices=args.p_slices,
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
    if args.mixup > 0.0 and args.mode != "regression":
        print("Mixup Activated!")
        mixup = MixUp(alpha=args.mixup)

    optimizer = select_optimizer(
        model=model,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if mixup is not None:
        assert args.mode != "regression", "Can't add MixUp with regression labels"
        if args.mode == "binary":
            loss_fn = mixup_criterion(criterion=torch.nn.BCEWithLogitsLoss())
        else:
            loss_fn = mixup_criterion(criterion=torch.nn.CrossEntropyLoss())
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
    if args.mode == "ssl":
        _ = train_vit_bt(
            model=model,
            model_without_ddp=model_without_ddp,
            distributed=args.distributed,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            epochs=args.epochs,
            patience=args.patience,
            accum_iter=args.accum_iter,
            target_dir=args.out_dir,
            model_name=args.model_name,
            device=args.gpu,
            verbose=args.verbose,
        )
    else:
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
            device=args.gpu,
            verbose=args.verbose,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="NiChart_ViT",
        add_help=False,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    mae_train = subparsers.add_parser(
        "mae_pretrain", help="Select MAE pretraining mode"
    )
    mae_train.add_argument(
        "--img_size",
        type=int,
        default=128,
        required=False,
    )

    mae_train.add_argument(
        "--masking_type",
        type=str,
        default="random",
        required=False,
    )

    mae_train.add_argument(
        "--mask_ratio",
        type=float,
        default=0.75,
        required=False,
    )

    mae_train.add_argument(
        "--model",
        type=str,
        default="mae_base",
        required=False,
    )

    mae_train.add_argument(
        "--in_dir",
        type=str,
        required=True,
    )

    mae_train.add_argument(
        "--batch_size",
        type=int,
        default=32,
        required=False,
    )

    mae_train.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default=".",
        required=False,
    )

    mae_train.add_argument(
        "--model_name",
        type=str,
        default="NiChart_MAE_ViT_best.pth",
        required=False,
    )

    mae_train.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda",
        required=False,
    )

    mae_train.add_argument(
        "--lr",
        type=float,
        default=0.01,
        required=False,
    )

    mae_train.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        required=False,
    )

    mae_train.add_argument(
        "--epochs",
        type=int,
        default=200,
        required=False,
    )

    mae_train.add_argument(
        "--patience",
        type=int,
        default=10,
        required=False,
    )

    mae_train.add_argument(
        "--optimizer",
        type=str,
        default="Adadelta",
        required=False,
    )

    mae_train.add_argument(
        "--accum_iter",
        type=int,
        default=1,
        required=False,
    )

    mae_train.add_argument(
        "--verbose",
        action="store_true",
    )

    mae_train.add_argument("--distributed", action="store_true")

    mae_train.add_argument(
        "--world_size",
        type=int,
        default=1,
        required=False,
    )

    mae_train.add_argument(
        "--dist_url",
        default="env://",
    )
    mae_train.set_defaults(func=run_mae_pretrain)

    vit_finetune = subparsers.add_parser(
        "vit_finetune", help="Select MAE pretraining mode"
    )
    vit_finetune.add_argument(
        "--img_size",
        type=str,
        default="128-128-128",
        required=False,
    )

    vit_finetune.add_argument(
        "--patch_size",
        type=str,
        default="8-8-8",
        required=False,
    )

    vit_finetune.add_argument(
        "--p_slices",
        type=float,
        default=1.0,
        required=False,
    )

    vit_finetune.add_argument(
        "--mode",
        type=str,
        required=True,
        help="Either regression, binary, multiclass or ssl",
    )

    vit_finetune.add_argument(
        "--projector",
        type=str,
        default="1024-2048-4096",
        required=False,
        help="Only if mode==ssl. Provide MLP projector dimensions for Barlow Twins loss",
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
        default="NiChart_ViT_best.pth",
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
        "--lambd",
        type=float,
        default=5e-3,
        required=False,
        help="Only for ViT SSL with barlow twins. Set's the lambda of barlow twins loss",
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

    # augmentation specifics
    vit_finetune.add_argument(
        "--smoothing",
        type=float,
        default=0.1,
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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
    destroy_process_group()
