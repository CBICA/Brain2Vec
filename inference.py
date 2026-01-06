import nibabel as nib
import torch
import torchio as tio
from torch.nn import functional as F

from NiChart_ViT.models.mae import mae_vit_base_patch8_dec512d8b
from NiChart_ViT.trainer.utils import load_weights


def run_one_scan(scan: str, weights: str) -> None:
    model = mae_vit_base_patch8_dec512d8b(img_size=128)
    load_weights(model, weights, "cpu")
    model.eval()

    niftii_img = nib.load(scan)
    img_header = niftii_img.header.copy()
    original_dims = img_header.get_data_shape()

    _img = tio.ScalarImage(scan)
    resize = tio.Resize((128, 128, 128))
    img_resized = resize(_img)

    original_mean = img_resized.data.float().mean()
    original_std = img_resized.data.float().std()

    normalize = tio.ZNormalization(masking_method=None)
    img_normalized = normalize(img_resized)
    img_tensor = img_normalized.data.float()

    with torch.no_grad():
        _, pred, mask = model(img_tensor.unsqueeze(0), mask_ratio=0.75)

    y = model.unpatchify(pred)  # [1, 1, 128, 128, 128]

    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 3)
    mask = model.unpatchify(mask)

    x = img_tensor.unsqueeze(0)

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

    rec_to_img = nib.Nifti1Image(result, niftii_img.affine, header=img_header)
    nib.save(rec_to_img, "mae_test.nii.gz")

    print("Saved reconstruction + visible!")


if __name__ == "__main__":
    run_one_scan(
        scan="../Datasets/BAScores/ISTAG_CN_All/istaging_2_new/CN_split/eval/3419196_2_0_T1_LPS_dlicv_aligned.nii.gz",
        weights="weights/MAE_TEST.pth",
    )
