# Copyright 2026 CBICA
#
# Author: Spiros Maggioros
#
# Licenced under CBICA's non-Commercial Software Agreement:
# https://www.med.upenn.edu/cbica/software-agreement-non-commercial.html

import argparse
import json
import os
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from Brain2Vec.dataloader.loader import NiftiLoader
from Brain2Vec.models.brain2vec import (
    brain2vec_base_patch8_dec512d8b,
    brain2vec_huge_patch8_dec512d8b,
    brain2vec_large_patch8_dec512d8b,
)
from Brain2Vec.models.vit import vit_base_patch8, vit_huge_patch8, vit_large_patch8
from Brain2Vec.trainer.utils import load_weights


def brain2vec_models(img_size: int, device: str = "cuda") -> dict:
    return {
        "brain2vec_base": brain2vec_base_patch8_dec512d8b(img_size=img_size).to(device),
        "brain2vec_large": brain2vec_large_patch8_dec512d8b(img_size=img_size).to(
            device
        ),
        "brain2vec_huge": brain2vec_huge_patch8_dec512d8b(img_size=img_size).to(device),
    }


def vit_models(img_size: int, num_classes: int, device: str = "cuda") -> dict:
    return {
        "vit_base": (
            vit_base_patch8(img_size=img_size, num_classes=num_classes).to(device)
        ),
        "vit_large": (
            vit_large_patch8(img_size=img_size, num_classes=num_classes).to(device)
        ),
        "vit_huge": (
            vit_huge_patch8(img_size=img_size, num_classes=num_classes).to(device)
        ),
    }


def test_dataloader(in_dir: str) -> DataLoader:
    single_loader = NiftiLoader(
        mode="inference",
        in_dir=in_dir,
        augmentation=False,
    )
    dataloader = DataLoader(
        single_loader,
        batch_size=1,
        shuffle=False,
        num_workers=os.cpu_count() // 2,  # type: ignore
    )

    return dataloader


@torch.no_grad()  # type: ignore
def run_brain2vec_inference(args: Any) -> None:
    torch.manual_seed(42)
    if args.device == "cuda":
        torch.cuda.manual_seed(42)
    model = brain2vec_models(args.img_size, args.device)[args.model]
    load_weights(model, args.model_weights, args.device)

    dataloader = test_dataloader(args.in_dir)

    model.to(args.device)
    model.eval()
    out_features = {}

    for _, (imgs, mrids) in enumerate(dataloader):
        imgs = imgs.to(args.device)
        imgs = imgs.float()

        with torch.autocast(device_type=args.device):
            features, _, _ = model.forward_encoder(imgs, mask_ratio=0.0)
            features = features[:, 0, :].float()
            for i in range(len(mrids)):
                out_features[mrids[i]] = features[i].cpu().tolist()

    with open(args.out_json, "w") as f:
        json.dump(out_features, f)

    print(f"Inference results saved to {args.out_json} successfully!")


@torch.no_grad()  # type: ignore
def run_vit_inference(args: Any) -> None:
    torch.manual_seed(42)
    model = vit_models(args.img_size, int(args.num_classes), args.device)[args.model]
    load_weights(model, args.model_weights, args.device)

    if args.out_json == "":
        args.out_json = None

    dataloader = test_dataloader(args.in_dir)

    model.to(args.device)
    model.eval()
    y_preds = []
    out_features = []

    for _, (imgs, mrids) in enumerate(dataloader):
        imgs = imgs.to(args.device)
        imgs = imgs.float()

        with torch.autocast(device_type=args.device):
            if args.out_json:
                _features = model.forward_features(imgs)
                _features = _features[:, 0, :].float()
                _features = _features.cpu().tolist()
                out_features.extend(_features)
            y_pred = model(imgs)

        if args.mode == "binary":
            y_pred = (torch.sigmoid(y_pred) > 0.5).float()
        if args.mode == "multiclass":
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

    if args.out_json:
        assert len(out_features) != 0
        with open(args.out_json, "w") as f:
            json.dump(out_features, f, indent=4)
        print(f"Embeddings saved to {args.out_json} successfully!")

    output_dir = os.path.dirname(args.out_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    inference_res.to_csv(args.out_csv, index=False)
    print(f"Inference results saved to {args.out_csv} successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="inference",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    brain2vec_inference = subparsers.add_parser(
        "brain2vec_inference", help="Select brain2vec inference"
    )

    brain2vec_inference.add_argument(
        "--img_size",
        type=int,
        required=True,
        help="Image input size. We usually resize to (128, 128, 128) so --img_size=128",
    )

    brain2vec_inference.add_argument(
        "--model",
        type=str,
        required=True,
        help="Either brain2vec_base, brain2vec_large or brain2vec_huge",
    )

    brain2vec_inference.add_argument(
        "--model_weights",
        type=str,
        required=True,
        help="Path to brain2vec model weights",
    )

    brain2vec_inference.add_argument(
        "--in_dir",
        type=str,
        required=True,
        help="Input directory with niftii images to perform inference on",
    )

    brain2vec_inference.add_argument(
        "--out_json",
        type=str,
        required=True,
        help="Name of output json file that output embeddings will be saved",
    )

    brain2vec_inference.add_argument(
        "--device",
        type=str,
        default="cuda",
        required=False,
        help="Either cuda, cpu or mps. Note that mps do not support some ops",
    )
    brain2vec_inference.set_defaults(func=run_brain2vec_inference)

    vit_inference = subparsers.add_parser(
        "vit_inference", help="Select downstream vit inference"
    )

    vit_inference.add_argument(
        "--model",
        type=str,
        required=True,
        help="Either vit_base, vit_large or vit_huge",
    )

    vit_inference.add_argument(
        "--img_size",
        type=int,
        required=True,
        help="Image input size. We usually resize to (128, 128, 128) so --img_size=128",
    )

    vit_inference.add_argument(
        "--mode",
        type=str,
        required=True,
        help="Either regression, binary or multiclass",
    )

    vit_inference.add_argument(
        "--num_classes",
        type=int,
        required=False,
        default=1,
        help="Specify the number of classes for multiclass problems. Default = 1",
    )

    vit_inference.add_argument(
        "--model_weights",
        type=str,
        required=True,
        help="Path to brain2vec model weights",
    )

    vit_inference.add_argument(
        "--in_dir",
        type=str,
        required=True,
        help="Input directory with niftii images to perform inference on",
    )

    vit_inference.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Name of output csv file with predictions",
    )

    vit_inference.add_argument(
        "--out_json",
        type=str,
        default="",
        required=False,
        help="If provided, output features will be saved in this json file",
    )

    vit_inference.add_argument(
        "--device",
        type=str,
        default="cuda",
        required=False,
        help="Either cuda, cpu or mps. Note that mps do not support some ops",
    )
    vit_inference.set_defaults(func=run_vit_inference)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
