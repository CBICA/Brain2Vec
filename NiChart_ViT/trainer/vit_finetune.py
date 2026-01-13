from typing import Optional

import torch
from timm.data import Mixup
from torch.utils.data import DataLoader
from torchmetrics import (  # AUROC,
    Accuracy,
    ConfusionMatrix,
    F1Score,
    MeanSquaredError,
    NormalizedRootMeanSquaredError,
    Precision,
    R2Score,
    Recall,
    Specificity,
)
from tqdm.auto import tqdm
from typing_extensions import Literal

from NiChart_ViT.trainer.utils import EarlyStopper, all_reduce_mean, save_model


def train_step(
    model: torch.nn.Module,
    mode: str,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    mixup_fn: Optional[Mixup],
    num_classes: Optional[int] = None,
    device: Literal["cuda", "mps", "cpu"] = "cuda",
    distributed: bool = False,
) -> dict:
    model.train(True)

    if mode == "regression":
        assert num_classes is None
        stats = {
            "train_mae": 0.0,
            "train_mse": 0.0,
            "train_nrmse": 0.0,
            "train_r2": 0.0,
        }

        mse = MeanSquaredError().to(device)
        nrmse = NormalizedRootMeanSquaredError().to(device)
        r2_score = R2Score().to(device)
    else:
        stats = {
            "train_acc": 0.0,
            "train_auc": 0.0,
            "train_recall": 0.0,
            "train_precision": 0.0,
            "train_specificity": 0.0,
            "train_f1": 0.0,
        }

        if mode == "multiclass":
            assert num_classes is not None
            acc = Accuracy(task=mode, average="macro", num_classes=num_classes).to(
                device
            )
            # auc = AUROC(task=mode, average="macro", num_classes=num_classes).to(device)
            recall = Recall(task=mode, average="macro", num_classes=num_classes).to(
                device
            )
            precision = Precision(
                task=mode, average="macro", num_classes=num_classes
            ).to(device)
            specificity = Specificity(
                task=mode, average="macro", num_classes=num_classes
            ).to(device)
            f1_score = F1Score(task=mode, average="macro", num_classes=num_classes).to(
                device
            )
        else:
            acc = Accuracy(task=mode).to(device)
            # auc = AUROC(task=mode).to(device)
            recall = Recall(task=mode).to(device)
            precision = Precision(task=mode).to(device)
            specificity = Specificity(task=mode).to(device)
            f1_score = F1Score(task=mode).to(device)

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y_original = y.clone()

        if mode != "multiclass":
            X, y = X.float(), y.float()
        else:
            X, y = X.float(), y.long()

        if mixup_fn is not None:
            X, y = mixup_fn(X, y)

        optimizer.zero_grad()

        # make prediction
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y_pred = model(X)
            if mode != "multiclass":
                y_pred = y_pred.squeeze()

            # loss computation
            loss = loss_fn(y_pred, y)

        loss.backward()

        optimizer.step()

        torch.cuda.synchronize()

        if mode == "regression":
            stats["train_mae"] += loss.item()
            mse.update(y_pred, y)
            nrmse.update(y_pred, y)
            r2_score.update(y_pred, y)
        else:
            if mode == "binary":
                y_pred_classes = torch.sigmoid(y_pred > 0.5).float()
            else:
                _, y_pred_classes = y_pred.topk(1)
                y_pred_classes = y_pred_classes.squeeze(dim=-1)

            acc.update(y_pred_classes, y_original)
            # auc.update(y_pred, y)
            recall.update(y_pred_classes, y_original)
            precision.update(y_pred_classes, y_original)
            specificity.update(y_pred_classes, y_original)
            f1_score.update(y_pred_classes, y_original)

    if mode == "regression":
        stats["train_mae"] = stats["train_mae"] / float(len(dataloader))
        stats["train_mse"] = mse.compute().item()
        stats["train_nrmse"] = nrmse.compute().item()
        stats["train_r2"] = r2_score.compute().item()
        if distributed:
            stats["train_mae"] = all_reduce_mean(stats["train_mae"])
    else:
        stats["train_acc"] = acc.compute().item()
        # stats["train_auc"] = auc.compute().item()
        stats["train_recall"] = recall.compute().item()
        stats["train_precision"] = precision.compute().item()
        stats["train_specificity"] = specificity.compute().item()
        stats["train_f1"] = f1_score.compute().item()

    return stats


def test_step(
    model: torch.nn.Module,
    mode: str,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    num_classes: Optional[int] = None,
    device: Literal["cuda", "mps", "cpu"] = "cuda",
    distributed: bool = False,
) -> dict:
    # eval mode
    model.eval()
    if mode == "regression":
        assert num_classes is None
        stats = {"test_mae": 0.0, "test_mse": 0.0, "test_nrmse": 0.0, "test_r2": 0.0}

        mse = MeanSquaredError().to(device)
        nrmse = NormalizedRootMeanSquaredError().to(device)
        r2_score = R2Score().to(device)
    else:
        stats = {
            "test_acc": 0.0,
            "test_auc": 0.0,
            "test_recall": 0.0,
            "test_precision": 0.0,
            "test_specificity": 0.0,
            "test_f1": 0.0,
            "test_confussion_matrix": [],  # type: ignore
        }

        if mode == "multiclass":
            assert num_classes is not None
            acc = Accuracy(task=mode, average="macro", num_classes=num_classes).to(
                device
            )
            # auc = AUROC(task=mode, average="macro", num_classes=num_classes).to(device)
            recall = Recall(task=mode, average="macro", num_classes=num_classes).to(
                device
            )
            precision = Precision(
                task=mode, average="macro", num_classes=num_classes
            ).to(device)
            specificity = Specificity(
                task=mode, average="macro", num_classes=num_classes
            ).to(device)
            f1_score = F1Score(task=mode, average="macro", num_classes=num_classes).to(
                device
            )
            confmat = ConfusionMatrix(task=mode, num_classes=num_classes).to(device)
        else:
            acc = Accuracy(task=mode).to(device)
            # auc = AUROC(task=mode).to(device)
            recall = Recall(task=mode).to(device)
            precision = Precision(task=mode).to(device)
            specificity = Specificity(task=mode).to(device)
            f1_score = F1Score(task=mode).to(device)
            confmat = ConfusionMatrix(task=mode).to(device)

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            if mode != "multiclass":
                X, y = X.float(), y.float()
            else:
                X, y = X.float(), y.long()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                test_pred = model(X)
                if mode != "multiclass":
                    test_pred = test_pred.squeeze()
                if mode == "regression":
                    loss = loss_fn(test_pred, y)

            if mode == "regression":
                stats["test_mae"] += loss.item()
                mse.update(test_pred, y)
                nrmse.update(test_pred, y)
                r2_score.update(test_pred, y)
            else:
                if mode == "binary":
                    y_pred_classes = (torch.sigmoid(test_pred) > 0.5).float()
                else:
                    _, y_pred_classes = test_pred.topk(1)
                    y_pred_classes = y_pred_classes.squeeze(dim=-1)

                acc.update(y_pred_classes, y)
                # auc.update(test_pred, y)
                recall.update(y_pred_classes, y)
                precision.update(y_pred_classes, y)
                specificity.update(y_pred_classes, y)
                f1_score.update(y_pred_classes, y)
                confmat.update(y_pred_classes, y)

    if mode == "regression":
        stats["test_mae"] = stats["test_mae"] / float(len(dataloader))
        stats["test_mse"] = mse.compute().item()
        stats["test_nrmse"] = nrmse.compute().item()
        stats["test_r2"] = r2_score.compute().item()
        if distributed:
            stats["test_mae"] = all_reduce_mean(stats["test_mae"])
    else:
        stats["test_acc"] = acc.compute().item()
        # stats["test_auc"] = auc.compute().item()
        stats["test_recall"] = recall.compute().item()
        stats["test_precision"] = precision.compute().item()
        stats["test_specificity"] = specificity.compute().item()
        stats["test_f1"] = f1_score.compute().item()

    return stats


def train(
    model: torch.nn.Module,
    model_without_ddp: torch.nn.Module,
    distributed: bool,
    mode: str,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    mixup_fn: Optional[Mixup],
    loss_fn: torch.nn.Module,
    epochs: int,
    patience: int,
    num_classes: Optional[int] = None,
    device: Literal["cuda", "mps", "cpu"] = "cuda",
    target_dir: str = ".",
    model_name: str = "NiChart_BAScores_best.pth",
    verbose: bool = False,
) -> dict:

    if mode == "regression":
        assert num_classes is None
        results: dict = {
            "train_mae": [],
            "train_mse": [],
            "train_nrmse": [],
            "train_r2": [],
            "test_mae": [],
            "test_mse": [],
            "test_nrmse": [],
            "test_r2": [],
        }
        best_res = float("inf")
    else:
        if mode == "multiclass":
            assert num_classes is not None
        results = {
            "train_acc": [],
            "train_auc": [],
            "train_recall": [],
            "train_precision": [],
            "train_specificity": [],
            "train_f1": [],
            "test_acc": [],
            "test_auc": [],
            "test_recall": [],
            "test_precision": [],
            "test_specificity": [],
            "test_f1": [],
            "test_confussion_matrix": [],
        }
        best_res = 0.0

    early_stopper = EarlyStopper(
        patience=patience, increase=True if (mode != "regression") else False
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in tqdm(range(epochs)):
        if distributed:
            train_dataloader.sampler.set_epoch(epoch)
            test_dataloader.sampler.set_epoch(epoch)

        train_stats = train_step(
            model=model,
            mode=mode,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            mixup_fn=mixup_fn,
            device=device,
            num_classes=num_classes if num_classes is not None else None,
            distributed=distributed,
        )

        test_stats = test_step(
            model=model,
            mode=mode,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
            num_classes=num_classes if num_classes is not None else None,
            distributed=distributed,
        )

        lr_scheduler.step()

        if mode == "regression":
            if test_stats["test_mae"] < best_res:
                best_res = test_stats["test_mae"]
                save_model(model_without_ddp, target_dir, model_name)
                if verbose:
                    print(
                        f"[INFO] New best model saved at {target_dir} with test MAE: {best_res:.4f}"
                    )
            if early_stopper.early_stop(test_stats["test_mae"]):
                if verbose:
                    print("Early stopping the training!")
                break
        else:
            if test_stats["test_acc"] > best_res:
                best_res = test_stats["test_acc"]
                save_model(model_without_ddp, target_dir, model_name)
                if verbose:
                    print(
                        f"[INFO] New best model saved at {target_dir} with test accuracy: {best_res:.4f}"
                    )
            if early_stopper.early_stop(test_stats["test_acc"]):
                if verbose:
                    print("Early stopping the training!")
                break

        if mode == "regression":
            if verbose:
                print(
                    f"Epoch: {epoch+1} | "
                    f"train_mae: {train_stats['train_mae']:.4f} | "
                    f"train_mse: {train_stats['train_mse']:.4f} | "
                    f"train_nrmse: {train_stats['train_nrmse']:.4f} | "
                    f"train_r2: {train_stats['train_r2']:.4f} | "
                    f"test_mae: {test_stats['test_mae']:.4f} | "
                    f"test_mse: {test_stats['test_mse']:.4f} | "
                    f"test_nrmse: {test_stats['test_nrmse']:.4f} | "
                    f"test_r2: {test_stats['test_r2']:.4f}"
                )

            results["train_mae"].append(train_stats["train_mae"])
            results["train_mse"].append(train_stats["train_mse"])
            results["train_nrmse"].append(train_stats["train_nrmse"])
            results["train_r2"].append(train_stats["train_r2"])
            results["test_mae"].append(test_stats["test_mae"])
            results["test_mse"].append(test_stats["test_mse"])
            results["test_nrmse"].append(test_stats["test_nrmse"])
            results["test_r2"].append(test_stats["test_r2"])
        else:
            if verbose:
                print(
                    f"Epoch: {epoch+1} | "
                    f"train_acc: {train_stats['train_acc']:.4f} | "
                    f"train_auc: {train_stats['train_auc']:.4f} | "
                    f"train_recall: {train_stats['train_recall']:.4f} | "
                    f"train_precision: {train_stats['train_precision']:.4f} | "
                    f"train_specificity: {train_stats['train_specificity']:.4f} | "
                    f"train_f1: {train_stats['train_f1']:.4f} | "
                    f"test_acc: {test_stats['test_acc']:.4f} | "
                    f"test_auc: {test_stats['test_auc']:.4f} | "
                    f"test_recall: {test_stats['test_recall']:.4f} | "
                    f"test_precision: {test_stats['test_precision']:.4f} | "
                    f"test_specificity: {test_stats['test_specificity']:.4f} | "
                    f"test_f1: {test_stats['test_f1']:.4f} | "
                    f"test_confussion_matrix: {test_stats['test_confussion_matrix']}"
                )

            results["train_acc"].append(train_stats["train_acc"])
            results["train_auc"].append(train_stats["train_auc"])
            results["train_recall"].append(train_stats["train_recall"])
            results["train_precision"].append(train_stats["train_precision"])
            results["train_specificity"].append(train_stats["train_specificity"])
            results["train_f1"].append(train_stats["train_f1"])
            results["test_acc"].append(test_stats["test_acc"])
            results["test_auc"].append(test_stats["test_auc"])
            results["test_recall"].append(test_stats["test_recall"])
            results["test_precision"].append(test_stats["test_precision"])
            results["test_specificity"].append(test_stats["test_specificity"])
            results["test_f1"].append(test_stats["test_f1"])

    return results
