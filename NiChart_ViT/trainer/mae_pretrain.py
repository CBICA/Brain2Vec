import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from NiChart_ViT.trainer.utils import EarlyStopper, save_model


def train_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: str,
) -> float:
    model.train(True)

    mse = 0.0

    for X, _ in dataloader:
        X = X.to(device)
        X = X.float()

        optimizer.zero_grad()

        with torch.autocast(device_type=device, dtype=torch.float16):
            loss, _, _ = model(X)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mse += loss.item()

    mse /= float(len(dataloader))
    return mse


def test_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
) -> float:
    model.eval()

    mse = 0.0

    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            X = X.float()

            with torch.autocast(device_type=device, dtype=torch.float16):
                loss, _, _ = model(X)

            mse += loss.item()

    mse /= float(len(dataloader))
    return mse


def train(
    model: torch.nn.Module,
    model_without_ddp: torch.nn.Module,
    distributed: bool,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    patience: int,
    device: str,
    target_dir: str = ".",
    model_name: str = "NiChart_ViTMAE_pretrain.pth",
    verbose: bool = False,
) -> dict:
    results: dict = {
        "train_mse": [],
        "test_mse": [],
    }
    best_res = float("inf")

    early_stopper = EarlyStopper(
        patience=patience,
        increase=False,
    )

    scaler = torch.amp.GradScaler(device)

    for epoch in tqdm(range(epochs)):
        if distributed:
            train_dataloader.sampler.set_epoch(epoch)
            test_dataloader.sampler.set_epoch(epoch)

        train_mse = train_step(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )

        test_mse = test_step(
            model=model,
            dataloader=test_dataloader,
            device=device,
        )

        if test_mse < best_res:
            best_res = test_mse
            save_model(model_without_ddp, target_dir, model_name)
            if verbose:
                print(
                    f"[INFO] New best model saved at {target_dir} with test MSE: {best_res:.4f}",
                )
            if early_stopper.early_stop(test_mse):
                if verbose:
                    print("Early stopping the training")
                break

        if verbose:
            print(
                f"Epoch: {epoch + 1} | "
                f"train_mse: {train_mse:.4f} | "
                f"test_mse: {test_mse:.4f}"
            )

        results["train_mse"].append(train_mse)
        results["test_mse"].append(test_mse)

    return results
