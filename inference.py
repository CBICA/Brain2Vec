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

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Brain2Vec.dataloader.loader import NiftiLoader
from Brain2Vec.models.brain2vec import (
    brain2vec_base_patch8_dec512d8b,
    brain2vec_huge_patch8_dec512d8b,
    brain2vec_large_patch8_dec512d8b,
)
from Brain2Vec.models.utils import load_brain2vec
from Brain2Vec.models.vit import vit_base_patch8, vit_huge_patch8, vit_large_patch8
from Brain2Vec.trainer.utils import load_weights


def brain2vec_models() -> dict:
    return {
        "brain2vec_base": brain2vec_base_patch8_dec512d8b,
        "brain2vec_large": brain2vec_large_patch8_dec512d8b,
        "brain2vec_huge": brain2vec_huge_patch8_dec512d8b,
    }


def vit_models() -> dict:
    return {
        "vit_base": vit_base_patch8,
        "vit_large": vit_large_patch8,
        "vit_huge": vit_huge_patch8,
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

    model_class = brain2vec_models()[args.model]
    if len(args.model_weights) == 0:
        model = load_brain2vec(
            repo_id="cbica/Brain2Vec",
            weights_filename="brain2vec.safetensors",
            model_class=model_class,
            model_kwargs={"img_size": args.img_size},
            device=args.device,
        )
    else:
        model = model_class(img_size=args.img_size)
        load_weights(model, args.model_weights, args.device)

    dataloader = test_dataloader(args.in_dir)

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
    model_class = vit_models()[args.model]
    if len(args.model_weights) == 0:
        model = load_brain2vec(
            repo_id="cbica/Brain2Vec",
            weights_filename="vit_ba_finetuned.safetensors",
            model_class=model_class,
            model_kwargs={
                "img_size": args.img_size,
                "num_classes": int(args.num_classes),
            },
            device=args.device,
        )
    else:
        model = model_class(img_size=args.img_size, num_classes=int(args.num_classes))
        load_weights(model, args.model_weights, args.device)

    if args.out_json == "":
        args.out_json = None

    dataloader = test_dataloader(args.in_dir)

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

    output_dir = os.path.dirname(args.ourun_t_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    inference_res.to_csv(args.out_csv, index=False)
    print(f"Inference results saved to {args.out_csv} successfully!")


@torch.no_grad()  # type: ignore
def run_visualize_attention(args: Any) -> None:
    torch.manual_seed(42)
    device = torch.device(args.device)
    model_class = brain2vec_models()[args.model]
    if len(args.model_weights) == 0:
        model = load_brain2vec(
            repo_id="cbica/Brain2Vec",
            weights_filename="brain2vec.safetensors",
            model_class=model_class,
            model_kwargs={"img_size": args.img_size},
            device=args.device,
        )
    else:
        model = model_class(img_size=args.img_size)
        load_weights(model, args.model_weights, args.device)
    model.eval()
    model.to(device)

    os.makedirs(args.output_dir, exist_ok=True)
    patch_size = model.patch_embed.patch_size[0]
    dataloader = test_dataloader(args.in_dir)

    for idx, (imgs, mrids) in enumerate(dataloader):
        imgs = imgs.to(device).float()  # Shape: (1, 1, D, H, W)
        mrid = mrids[0] if isinstance(mrids, (list, tuple)) else mrids
        B, C, D, H, W = imgs.shape
        grid_d, grid_h, grid_w = D // patch_size, H // patch_size, W // patch_size

        attentions = model.get_last_selfattention(imgs)
        num_heads = attentions.shape[1]
        cls_attn = attentions[0, :, 0, 1:]

        if args.threshold > 0:
            val, idxs = torch.sort(cls_attn, dim=1)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval > (1 - args.threshold)
            idx2 = torch.argsort(idxs, dim=1)
            for head in range(num_heads):
                th_attn[head] = th_attn[head][idx2[head]]
            cls_attn = th_attn.float()

        attn_3d = cls_attn.reshape(1, num_heads, grid_d, grid_h, grid_w)
        attn_3d_upsampled = F.interpolate(
            attn_3d, size=(D, H, W), mode="trilinear", align_corners=False
        )[0]

        img_np = imgs[0, 0].cpu().numpy()
        attn_np = attn_3d_upsampled.cpu().numpy()

        axial_idx = D // 2
        coronal_idx = H // 2
        sagittal_idx = W // 2

        views = [
            ("Axial", img_np[axial_idx], attn_np[:, axial_idx]),
            ("Coronal", img_np[:, coronal_idx, :], attn_np[:, :, coronal_idx, :]),
            ("Sagittal", img_np[:, :, sagittal_idx], attn_np[:, :, :, sagittal_idx]),
        ]

        fig, axes = plt.subplots(
            3, num_heads + 1, figsize=((num_heads + 1) * 3, len(views) * 3)
        )
        for row, (view_name, img_slice, attn_slice) in enumerate(views):
            img_slice_rot = np.rot90(img_slice)
            axes[row, 0].imshow(img_slice_rot, cmap="gray")
            axes[row, 0].set_title(f"{view_name} - Original MRI")
            axes[row, 0].axis("off")
            for head in range(num_heads):
                attn_slice_rot = np.rot90(attn_slice[head])
                axes[row, head + 1].imshow(img_slice_rot, cmap="gray")
                axes[row, head + 1].imshow(attn_slice_rot, cmap="jet", alpha=0.5)
                axes[row, head + 1].set_title(f"Head {head}")
                axes[row, head + 1].axis("off")

        plot_path = os.path.join(args.output_dir, f"{mrid}_attention_heads.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()

        if getattr(args, "save_nifti", False):
            for head in range(num_heads):
                nifti_img = nib.Nifti1Image(attn_np[head], affine=np.eye(4))
                nib.save(
                    nifti_img,
                    os.path.join(args.output_dir, f"{mrid}_head_{head}.nii.gz"),
                )

        print(f"Attention map generated for {mrid} -> {plot_path}")


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
        required=False,
        default="",
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
        required=False,
        default="",
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

    attn_visualization = subparsers.add_parser(
        "visualize_attention", help="Select Brain2Vec attention visualization"
    )

    attn_visualization.add_argument(
        "--img_size",
        type=int,
        required=True,
        help="Image input size. We usually resize to (128, 128, 128) so --img_size=128",
    )

    attn_visualization.add_argument(
        "--model",
        type=str,
        required=True,
        help="Either brain2vec_base, brain2vec_large or brain2vec_huge",
    )

    attn_visualization.add_argument(
        "--in_dir",
        type=str,
        required=True,
        help="Input directory with niftii images to perform inference on",
    )

    attn_visualization.add_argument(
        "--model_weights",
        type=str,
        required=False,
        default="",
        help="Path to brain2vec model weights",
    )

    attn_visualization.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where attention heads will be saved",
    )

    attn_visualization.add_argument(
        "--threshold",
        type=float,
        required=False,
        default=0.0,
        help="Set to > 0 if you want to apply thresholding on output attention maps",
    )

    attn_visualization.add_argument(
        "--save_nifti",
        action="store_true",
        help="Set only if you want the output nifti attention heads to be saved",
    )

    attn_visualization.add_argument(
        "--device",
        type=str,
        required=False,
        help="Either cuda, cpu or mps. Note that mps do not support some ops",
    )

    attn_visualization.set_defaults(func=run_visualize_attention)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
