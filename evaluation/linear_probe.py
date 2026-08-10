
import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    """
    Simple linear classifier used for
    linear-probe evaluation.

    The encoder remains frozen.
    """

    def __init__(
        self,
        feature_dim=2048,
        num_classes=10
    ):

        super().__init__()

        self.classifier = nn.Linear(
            feature_dim,
            num_classes
        )


    def forward(self, x):

        return self.classifier(x)


def train_linear_probe(
    encoder,
    classifier,
    train_loader,
    optimizer,
    criterion,
    device,
    epochs=10
):
    """
    Train a linear classifier while keeping
    the encoder frozen.
    """

    encoder.eval()

    classifier.train()

    history = []

    for epoch in range(epochs):

        running_loss = 0.0

        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # No gradient through encoder
            with torch.no_grad():

                features = encoder(images)

            # Classifier forward pass
            outputs = classifier(features)

            loss = criterion(
                outputs,
                labels
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)


        epoch_loss = (
            running_loss /
            len(train_loader)
        )

        epoch_accuracy = (
            correct / total
        )

        history.append({
            "loss": epoch_loss,
            "accuracy": epoch_accuracy
        })

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {epoch_loss:.4f} "
            f"Accuracy: {epoch_accuracy:.4f}"
        )

    return history


def evaluate_linear_probe(
    encoder,
    classifier,
    test_loader,
    device
):
    """
    Evaluate the frozen encoder +
    trained linear classifier.
    """

    encoder.eval()
    classifier.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            features = encoder(images)

            outputs = classifier(
                features
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = correct / total

    return accuracy
