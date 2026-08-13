
import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import STL10

from data_utils.simclr_dataset import (
    SimCLRTransform,
    SimCLRDataset
)

from models.simclr import SimCLR
from losses.info_nce import InfoNCELoss

from configs.vit_simclr_config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    TEMPERATURE,
    IMAGE_SIZE,
    MAX_SAMPLES
)


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    stl10_dataset = STL10(
        root="/kaggle/working/data",
        split="train",
        download=True
    )

    stl10_dataset = Subset(
        stl10_dataset,
        range(min(MAX_SAMPLES, len(stl10_dataset)))
    )

    transform = SimCLRTransform(
        image_size=IMAGE_SIZE
    )

    simclr_dataset = SimCLRDataset(
        stl10_dataset,
        transform=transform
    )

    print(
        "Training samples:",
        len(simclr_dataset)
    )

    train_loader = DataLoader(
        simclr_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    model = SimCLR(
        encoder_type="vit",
        pretrained_encoder=True
    ).to(device)

    loss_fn = InfoNCELoss(
        temperature=TEMPERATURE
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    checkpoint_dir = os.path.join(
        PROJECT_ROOT,
        "checkpoints"
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )

    checkpoint_path = os.path.join(
        checkpoint_dir,
        "simclr_vit_stl10.pth"
    )

    loss_history = []

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        for images1, images2 in train_loader:

            images1 = images1.to(device)
            images2 = images2.to(device)

            optimizer.zero_grad()

            _, z1 = model(images1)
            _, z2 = model(images2)

            loss = loss_fn(z1, z2)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        epoch_loss = (
            running_loss / len(train_loader)
        )

        loss_history.append(epoch_loss)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {epoch_loss:.4f}"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
                "loss_history": loss_history
            },
            checkpoint_path
        )

        print("Checkpoint saved.")

    print("Training complete.")
    print("Final checkpoint:")
    print(checkpoint_path)


if __name__ == "__main__":
    main()
