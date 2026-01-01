import torch

from NiChart_ViT.dataloader.loader import create_dataloaders
from NiChart_ViT.models.vit import vit_base_patch16
from NiChart_ViT.trainer.utils import load_weights
from NiChart_ViT.trainer.vit_finetune import train


def main() -> None:
    model = vit_base_patch16(img_size=128, num_classes=1).to("cuda")
    load_weights(model=model, model_weights="weights/MAE_TEST.pth", device="cuda")

    train_loader, test_loader, _ = create_dataloaders(
        in_dir="../Datasets/BAScores/ISTAG_CN_All/istaging_2_new/CN_split",
        batch_size=32,
        label_dict_csv="../Datasets/BAScores/ISTAG_CN_All/istaging_2_new/istag_controlNoOverlap.csv",
        target="Age",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train(
        model=model,
        mode="regression",
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=2000,
        patience=20,
        device="cuda",
        target_dir="weights/",
        model_name="VIT_FINETUNE.pth",
        verbose=True,
    )


if __name__ == "__main__":
    main()
