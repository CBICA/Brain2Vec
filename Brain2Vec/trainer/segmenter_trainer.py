import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from Brain2Vec.trainer.utils import EarlyStopper
from Brain2Vec.trainer.utils import NativeScalerWithGradNormCount as NativeScaler
from Brain2Vec.trainer.utils import all_reduce_mean, save_model


def train_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_scaler: NativeScaler,
    accum_iter: int,
    device: str,
    distributed: bool = False,
) -> float:
    model.train(True)

    train_loss = 0.0
    num_updates = 0

    optimizer.zero_grad()

    for data_iter_step, (X, mask) in enumerate(dataloader):
        X, mask = X.to(device, non_blocking=True), mask.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model.forward(X, mask)

        loss_value = loss.item()
        loss /= accum_iter

        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        loss_value_reduce = all_reduce_mean(loss_value) if distributed else loss_value
        if (data_iter_step + 1) % accum_iter == 0:
            train_loss += loss_value_reduce
            num_updates += 1

    train_loss = train_loss / max(num_updates, 1)
    return train_loss


def test_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    distributed: bool = False,
) -> float:
    model.eval()

    test_loss = 0.0

    with torch.no_grad():
        for X, mask in dataloader:
            X, mask = X.to(device, non_blocking=True), mask.to(
                device, non_blocking=True
            )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model.forward(X, mask)

            test_loss += loss.item()

    test_loss /= float(len(dataloader))
    if distributed:
        test_loss = all_reduce_mean(test_loss)

    return test_loss


def train(
    model: torch.nn.Module,
    model_without_ddp: torch.nn.Module,
    distributed: bool,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    patience: int,
    accum_iter: int,
    device: str,
    target_dir: str = ".",
    model_name: str = "segmenter_best.pth",
    verbose: bool = False,
) -> dict:

    results: dict = {
        "train_loss": [],
        "test_loss": [],
    }
    best_res = float("inf")

    early_stopper = EarlyStopper(
        patience=patience,
        increase=False,
    )

    loss_scaler = NativeScaler()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in tqdm(range(epochs)):
        if distributed:
            train_dataloader.sampler.set_epoch(epoch)
            test_dataloader.sampler.set_epoch(epoch)

        train_loss = train_step(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            accum_iter=accum_iter,
            device=device,
            distributed=distributed,
        )

        test_loss = test_step(
            model=model,
            dataloader=test_dataloader,
            device=device,
            distributed=distributed,
        )

        lr_scheduler.step()

        if test_loss < best_res:
            best_res = test_loss
            save_model(model_without_ddp, target_dir, model_name)
            if verbose:
                print(
                    f"[INFO] New best model saved at {target_dir} with test accuracy: {best_res:.4f}"
                )
        if early_stopper.early_stop(test_loss):
            if verbose:
                print("Early stopping the training!")
            break

        if verbose:
            print(
                f"Epoch: {epoch + 1} | "
                f"train_loss: {train_loss:.4f} | "
                f"test_loss: {test_loss:.4f}"
            )

        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)

    return results
