import json
import os
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from NiChart_ViT.dataloader.loader import NiftiLoader
from NiChart_ViT.trainer.utils import load_weights


def test_dataloader(in_dir: str) -> DataLoader:
    single_loader = NiftiLoader(
        mode="inference",
        in_dir=in_dir,
    )
    dataloader = DataLoader(
        single_loader,
        batch_size=1,
        shuffle=False,
        num_workers=os.cpu_count() // 2,  # type: ignore
    )

    return dataloader


@torch.no_grad()  # type: ignore
def mae_inference(
    model: torch.nn.Module,
    model_weights: str,
    in_dir: str,
    out_json: str,
    device: str = "cuda",
) -> None:
    load_weights(model, model_weights, device)

    dataloader = test_dataloader(in_dir)

    model.to(device)
    model.eval()
    out_features = {}

    for _, (imgs, mrids) in enumerate(dataloader):
        imgs = imgs.to(device)
        imgs = imgs.float()
        imgs.requires_grad_()

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            features, _, _ = model.forward_encoder(imgs)
            features = features[:, 0, :]
            for i in range(len(mrids)):
                out_features[mrids[i]] = features[i].cpu().tolist()

    with open(out_json, "w") as f:
        json.dump(out_features, f)

    print(f"Inference results saved to {out_json} successfully!")


@torch.no_grad()  # type: ignore
def vit_inference(
    model: torch.nn.Module,
    mode: str,
    model_weights: str,
    in_dir: str,
    out_csv: str,
    out_json: Optional[str] = None,
    device: str = "cuda",
) -> None:
    load_weights(model, model_weights, device)

    dataloader = test_dataloader(in_dir)

    model.to(device)
    model.eval()
    y_preds = []
    out_features = []

    for _, (imgs, mrids) in enumerate(dataloader):
        imgs = imgs.to(device)
        imgs = imgs.float()
        imgs.requires_grad_()

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            if out_json:
                _features = model.forward_features(imgs)
                _features = _features[:, 0, :]
                _features = _features.cpu().tolist()
                out_features.extend(_features)
            y_pred = model(imgs)

        if mode == "binary":
            y_pred = (torch.sigmoid(y_pred) > 0.5).float()
        if mode == "multiclass":
            _, y_pred = y_pred.topk(1)
            y_pred = y_pred.squeeze(dim=-1)

        y_pred = y_pred.cpu().tolist()
        if isinstance(mrids, (list, tuple)):
            for mrid, pred in zip(mrids, y_pred):
                y_preds.append((mrid, pred))
        else:
            y_preds.append((mrids, y_pred))

    inference_res = pd.DataFrame(
        {
            "MRID": [y_pred[0] for y_pred in y_preds],
            "Prediction": [y_pred[1] for y_pred in y_preds],
        }
    )

    if out_json:
        assert len(out_features) != 0
        with open(out_json, "w") as f:
            json.dump(out_features, f)

    output_dir = os.path.dirname(out_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    inference_res.to_csv(out_csv, index=False)
    print(f"Embeddings saved to {out_json} successfully!")
    print(f"Inference results saved to {out_csv} successfully!")
