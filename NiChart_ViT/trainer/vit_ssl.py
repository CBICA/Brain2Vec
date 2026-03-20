import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from NiChart_ViT.trainer.utils import EarlyStopper
from NiChart_ViT.trainer.utils import NativeScalerWithGradNormCount as NativeScaler
from NiChart_ViT.trainer.utils import all_reduce_mean, save_model


def train_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_scaler: NativeScaler,
    accum_iter: int,
    device: str,
    distributed: bool = False,
) -> float:
    model.train()

    optimizer.zero_grad()
    num_updates = 0
    total_loss = 0.0

    for data_iter_step, X in enumerate(dataloader):
        _, x_a, x_b = X
        x_a, x_b = (
            x_a.to(device, non_blocking=True),
            x_b.to(device, non_blocking=True),
        )

        x_a, x_b = x_a.float(), x_b.float()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x_a, x_b)

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
            total_loss += loss_value_reduce
            num_updates += 1

    total_loss = total_loss / max(num_updates, 1)
    return total_loss


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
    device: str = "cuda",
    target_dir: str = ".",
    model_name: str = "NiChart_BarlowTwins_best.pth",
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

        lr_scheduler.step()

        if train_loss < best_res:
            best_res = train_loss
            save_model(model_without_ddp, target_dir, model_name)
            if verbose:
                print(
                    f"[INFO] New best model saved at {target_dir} with test loss: {best_res:.4f}"
                )
            if early_stopper.early_stop(train_loss):
                if verbose:
                    print("Early stopping the training")
                break

        if verbose:
            print(
                f"Epoch: {epoch + 1} | " f"barlow twins loss: {train_loss:.4f} | ",
            )

        results["train_loss"].append(train_loss)

    return results
