import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from NiChart_ViT.trainer.utils import EarlyStopper
from NiChart_ViT.trainer.utils import NativeScalerWithGradNormCount as NativeScaler
from NiChart_ViT.trainer.utils import all_reduce_mean, barlow_twins_loss, save_model


def train_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    masking_type: str,
    loss_scaler: NativeScaler,
    mask_ratio: float,
    accum_iter: int,
    device: str,
    distributed: bool = False,
) -> float:
    model.train(True)

    optimizer.zero_grad()
    num_updates = 0
    total_loss = 0.0

    for data_iter_step, X in enumerate(dataloader):
        X, x_a, x_b, _ = X
        X, x_a, x_b = X.to(device), x_a.to(device), x_b.to(device)
        X, x_a, x_b = X.float(), x_a.float(), x_b.float()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            rec_loss, _, _ = model(
                X,
                masking_type=masking_type,
                mask_ratio=mask_ratio,
            )

            features_a, _, _ = model.module.forward_encoder(x_a)
            features_b, _, _ = model.module.forward_encoder(x_b)

            bt_loss = barlow_twins_loss(features_a[:, 0, :], features_b[:, 0, :])
            loss = rec_loss + 10 * bt_loss

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


def test_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    masking_type: str,
    mask_ratio: float,
    device: str,
    distributed: bool = False,
) -> float:
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for X in dataloader:
            X, x_a, x_b, _ = X
            X, x_a, x_b = X.to(device), x_a.to(device), x_b.to(device)
            X, x_a, x_b = X.float(), x_a.float(), x_b.float()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                rec_loss, _, _ = model(
                    X,
                    masking_type=masking_type,
                    mask_ratio=mask_ratio,
                )

                features_a, _, _ = model.module.forward_encoder(x_a)
                features_b, _, _ = model.module.forward_encoder(x_b)

                bt_loss = barlow_twins_loss(features_a[:, 0, :], features_b[:, 0, :])
                loss = rec_loss + 10 * bt_loss

            total_loss += loss.item()

    total_loss /= float(len(dataloader))
    if distributed:
        total_loss = all_reduce_mean(total_loss)

    return total_loss


def train(
    model: torch.nn.Module,
    model_without_ddp: torch.nn.Module,
    distributed: bool,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    masking_type: str,
    mask_ratio: float,
    epochs: int,
    patience: int,
    accum_iter: int,
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

    loss_scaler = NativeScaler()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in tqdm(range(epochs)):
        if distributed:
            train_dataloader.sampler.set_epoch(epoch)
            test_dataloader.sampler.set_epoch(epoch)

        train_mse = train_step(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            masking_type=masking_type,
            mask_ratio=mask_ratio,
            loss_scaler=loss_scaler,
            accum_iter=accum_iter,
            device=device,
            distributed=distributed,
        )

        test_mse = test_step(
            model=model,
            dataloader=test_dataloader,
            masking_type=masking_type,
            mask_ratio=mask_ratio,
            device=device,
            distributed=distributed,
        )

        lr_scheduler.step()

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
                f"test_mse: {test_mse:.4f}",
            )

        results["train_mse"].append(train_mse)
        results["test_mse"].append(test_mse)

    return results
