import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_ROOT = "/kaggle/working/data"
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.0003
IMAGE_SIZE = 96


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.STL10(
        root=DATA_ROOT,
        split="train",
        download=True,
        transform=transform
    )

    test_dataset = datasets.STL10(
        root=DATA_ROOT,
        split="test",
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        10
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_accuracy = 0.0

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        "resnet50_supervised_stl10.pth"
    )

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                predictions = outputs.argmax(dim=1)

                total += labels.size(0)

                correct += (
                    predictions == labels
                ).sum().item()

        accuracy = correct / total

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {running_loss / len(train_loader):.4f} "
            f"Test Accuracy: {accuracy:.4f}"
        )

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_accuracy": accuracy
                },
                checkpoint_path
            )

            print("Best checkpoint saved.")

    print("Training complete.")
    print("Best Test Accuracy:", best_accuracy)


if __name__ == "__main__":
    main()
