import argparse
from typing import Any, Optional

import torch
from torch import nn
from torch.distributed import destroy_process_group

from NiChart_ViT.dataloader.loader import create_dataloaders
from NiChart_ViT.models.mae import (
    mae_vit_base_patch16_dec512d8b,
    mae_vit_huge_patch14_dec512d8b,
    mae_vit_large_patch16_dec512d8b,
)
from NiChart_ViT.models.vit import vit_base_patch16, vit_huge_patch14, vit_large_patch16
from NiChart_ViT.trainer.mae_pretrain import train as mae_pretrain
from NiChart_ViT.trainer.utils import init_distributed_mode, load_weights
from NiChart_ViT.trainer.vit_finetune import train as vit_finetune


def available_models(
    img_size: int = 128,
    num_classes: Optional[int] = None,
    device: str = "cuda",
) -> dict:
    return {
        "mae_base": torch.compile(
            mae_vit_base_patch16_dec512d8b(img_size=img_size).to(device), dynamic=False
        ),
        "mae_large": torch.compile(
            mae_vit_large_patch16_dec512d8b(img_size=img_size).to(device), dynamic=False
        ),
        "mae_huge": torch.compile(
            mae_vit_huge_patch14_dec512d8b(img_size=img_size).to(device), dynamic=False
        ),
        "vit_base": torch.compile(
            (
                vit_base_patch16(img_size=img_size)
                if num_classes is None
                else vit_base_patch16(img_size=img_size, num_classes=num_classes).to(
                    device
                )
            ),
            dynamic=False,
        ),
        "vit_large": torch.compile(
            (
                vit_large_patch16(img_size=img_size)
                if num_classes is None
                else vit_large_patch16(img_size=img_size, num_classes=num_classes).to(
                    device
                )
            ),
            dynamic=False,
        ),
        "vit_huge": torch.compile(
            (
                vit_huge_patch14(img_size=img_size)
                if num_classes is None
                else vit_huge_patch14(img_size=img_size, num_classes=num_classes).to(
                    device
                )
            ),
            dynamic=False,
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
            amsgrad=True,
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
        in_dir=args.in_dir,
        batch_size=args.batch_size,
        label_dict_csv=args.label_dict,
        target="Age",
        distributed=args.distributed,
        num_tasks=num_tasks if args.distributed else None,
        global_rank=global_rank if args.distributed else None,
    )

    eff_batch_size = args.batch_size * args.world_size
    eff_lr = args.lr * eff_batch_size / 256

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
        optimizer=optimizer,
        epochs=args.epochs,
        patience=args.patience,
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

    avail_models = available_models(
        img_size=args.img_size,
        num_classes=1 if args.num_classes == -1 else args.num_classes,
        device=args.device,
    )
    assert args.model in avail_models.keys()

    init_distributed_mode(args)

    model = avail_models[args.model]
    load_weights(model=model, model_weights=args.pretrain, device=args.device)
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = model.to(args.device)

    model_without_ddp = model
    num_tasks = args.world_size
    global_rank = args.rank

    train_dataloader, test_dataloader, _ = create_dataloaders(
        in_dir=args.in_dir,
        batch_size=args.batch_size,
        label_dict_csv=args.label_dict,
        target=args.target,
        distributed=args.distributed,
        num_tasks=num_tasks if args.distributed else None,
        global_rank=global_rank if args.distributed else None,
    )

    eff_batch_size = args.batch_size * args.world_size
    eff_lr = args.lr * eff_batch_size / 256

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
        "--label_dict",
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
        required=True,
    )

    vit_finetune.add_argument(
        "--pretrain",
        type=str,
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
        required=True,
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
