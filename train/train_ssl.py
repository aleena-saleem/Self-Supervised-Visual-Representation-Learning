import sys
import os



PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import STL10

from data_utils.simclr_dataset import (
    SimCLRTransform,
    SimCLRDataset
)

from models.simclr import SimCLR
from losses.info_nce import InfoNCELoss

from configs.simclr_config import (
    DATA_ROOT,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    TEMPERATURE,
    IMAGE_SIZE,
    MAX_SAMPLES,
    CHECKPOINT_NAME,
    RESUME
)


def main():


    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)




    stl10_dataset = STL10(
        root=DATA_ROOT,
        split="train",
        download=True
    )

    stl10_dataset = Subset(
        stl10_dataset,
        range(
            min(
                MAX_SAMPLES,
                len(stl10_dataset)
            )
        )
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
        pin_memory=torch.cuda.is_available()
    )



    model = SimCLR(
        pretrained_encoder=True
    )

    model = model.to(device)


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
        CHECKPOINT_NAME
    )




    start_epoch = 0

    loss_history = []


  

    if RESUME and os.path.exists(
        checkpoint_path
    ):

        print()
        print("Loading checkpoint...")
        print(checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = checkpoint["epoch"]

        loss_history = checkpoint.get(
            "loss_history",
            []
        )

        print(
            "Resuming from epoch:",
            start_epoch
        )


    elif RESUME:

        print()
        print(
            "RESUME=True but checkpoint "
            "was not found."
        )

        print(
            "Starting training from epoch 1."
        )



    for epoch in range(
        start_epoch,
        EPOCHS
    ):

        model.train()

        running_loss = 0.0

        for view1, view2 in train_loader:

            view1 = view1.to(device)
            view2 = view2.to(device)

            # Forward pass
            _, z1 = model(view1)
            _, z2 = model(view2)

            # InfoNCE loss
            loss = loss_fn(
                z1,
                z2
            )

            # Backpropagation
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()


        average_loss = (
            running_loss /
            len(train_loader)
        )

        loss_history.append(
            average_loss
        )


        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {average_loss:.4f}"
        )



        checkpoint = {

            "epoch":
                epoch + 1,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "loss":
                average_loss,

            "loss_history":
                loss_history
        }

        torch.save(
            checkpoint,
            checkpoint_path
        )

        print(
            "Checkpoint saved."
        )


    results_dir = os.path.join(
        PROJECT_ROOT,
        "results"
    )

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    loss_plot_path = os.path.join(
        results_dir,
        "simclr_training_loss.png"
    )


    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        range(
            1,
            len(loss_history) + 1
        ),
        loss_history,
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("InfoNCE Loss")

    plt.title(
        "SimCLR Training Loss"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        loss_plot_path,
        dpi=200
    )

    plt.close()


    print()
    print(
        "Loss plot saved to:"
    )

    print(
        loss_plot_path
    )


    print()
    print(
        "Training complete!"
    )

    print(
        "Final checkpoint:"
    )

    print(
        checkpoint_path
    )


if __name__ == "__main__":
    main()
