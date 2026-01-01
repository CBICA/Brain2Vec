from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def get_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1

    return int(dist.get_world_size())


def get_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1

    return int(dist.get_rank())


def is_main_process() -> bool:
    return get_rank() == 0


def save_on_master(*args: Any, **kwargs: Any) -> None:
    if is_main_process():
        torch.save(*args, **kwargs)


class EarlyStopper:
    """
    Performs early stopping as PyTorch doesn't have an implemented one
    """

    def __init__(
        self, patience: int = 1, min_delta: float = 0.0, increase: bool = False
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.increase = increase
        self.best_value = float("-inf") if increase else float("inf")

    def early_stop(self, validation_val: float) -> bool:
        if not self.increase:
            if validation_val <= self.best_value:
                self.best_value = validation_val
                self.counter = 0
            else:
                self.counter += 1
        else:
            if validation_val >= self.best_value:
                self.best_value = validation_val
                self.counter = 0
            else:
                self.counter += 1

        return self.counter > self.patience


def save_model(model: torch.nn.Module, target_dir: str, model_name: str) -> None:

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def load_weights(model: torch.nn.Module, model_weights: str, device: str) -> None:
    state_dict = torch.load(model_weights, map_location=torch.device(device))
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    total_model_keys = len(model.state_dict().keys())
    loaded_keys = total_model_keys - len(missing_keys)

    print(f"Successfully loaded: {loaded_keys} | Total missing keys: {missing_keys}")
