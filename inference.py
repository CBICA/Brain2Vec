import os
from typing import Optional

import nibabel as nib
import pandas as pd
import torch
import torchio as tio
from torch.nn import functional as F
from torch.utils.data import DataLoader

from NiChart_ViT.dataloader.loader import NiftiLoader
from NiChart_ViT.models.mae import mae_vit_base_patch8_dec512d8b
from NiChart_ViT.trainer.utils import load_weights


def vit_inference(
    model: torch.nn.Module,
    mode: str,
    model_weights: str,
    in_dir: str,
    out_csv: str,
    device: str = "cuda",
) -> None:
    load_weights(model, model_weights, device)

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

    model.to(device)
    model.eval()
    y_preds = []

    for _, (imgs, mrids) in enumerate(dataloader):
        imgs = imgs.to(device)
        imgs = imgs.float()
        imgs.requires_grad_()

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            y_pred = model(imgs)

        if mode == "binary":
            y_pred_classes = (torch.sigmoid(y_pred) > 0.5).float()
        else:
            _, y_pred_classes = y_pred.topk(1)
            y_pred_classes = y_pred_classes.squeeze(dim=-1)

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

    output_dir = os.path.dirname(out_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    inference_res.to_csv(out_csv, index=False)


def run_one_scan(scan: str, weights: str, ground_truth: Optional[str] = None) -> None:
    model = mae_vit_base_patch8_dec512d8b(img_size=128)
    load_weights(model, weights, "cpu")
    model.eval()

    niftii_img = nib.load(scan)
    img_header = niftii_img.header.copy()
    original_dims = img_header.get_data_shape()

    _img = tio.ScalarImage(scan)
    resize = tio.Resize((128, 128, 128))
    normalize = tio.ZNormalization(masking_method=None)
    img_resized = resize(_img)

    if ground_truth is not None:
        mask_img = tio.ScalarImage(ground_truth)
        mask_img = resize(mask_img)
        # mask_img = normalize(mask_img)
        mask_img_tensor = mask_img.data.float()
        mask_img = torch.where(mask_img_tensor > 0, 1.0, 0.0)

    original_mean = img_resized.data.float().mean()
    original_std = img_resized.data.float().std()

    img_normalized = normalize(img_resized)
    img_tensor = img_normalized.data.float()

    with torch.no_grad():
        _, pred, mask = model(
            img_tensor.unsqueeze(0),
            masking_type="block",
            mask_ratio=0.05,
        )

    y = model.unpatchify(pred)  # [1, 1, 128, 128, 128]

    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 3)
    mask = model.unpatchify(mask)
    mask = mask.detach().cpu()

    x = img_tensor.unsqueeze(0)

    im_masked = x * (1 - mask)

    im_paste = x * (1 - mask) + y * mask

    im_paste = im_paste * original_std + original_mean

    im_paste_upsampled = F.interpolate(
        im_paste,
        size=original_dims,
        mode="trilinear",
        align_corners=False,
    )

    result = im_paste_upsampled.squeeze(0).squeeze(0)
    result = result.detach().cpu().numpy()
    im_masked = im_masked.squeeze(0).squeeze(0).detach().cpu().numpy()

    mask_to_img = nib.Nifti1Image(im_masked, niftii_img.affine, header=img_header)
    rec_to_img = nib.Nifti1Image(result, niftii_img.affine, header=img_header)
    nib.save(rec_to_img, "mae_test.nii.gz")
    nib.save(mask_to_img, "mask.nii.gz")

    print("Saved reconstruction + visible!")


if __name__ == "__main__":
    pass
