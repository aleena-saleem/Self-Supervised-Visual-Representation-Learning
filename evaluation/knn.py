
import torch
import torch.nn.functional as F


def extract_features(
    encoder,
    dataloader,
    device
):
    """
    Extract feature representations and labels
    from a dataset using the encoder.
    """

    encoder.eval()

    features = []
    labels = []

    with torch.no_grad():

        for images, batch_labels in dataloader:

            images = images.to(device)

            batch_features = encoder(images)

            # Normalize representations
            batch_features = F.normalize(
                batch_features,
                dim=1
            )

            features.append(
                batch_features.cpu()
            )

            labels.append(
                batch_labels.cpu()
            )

    features = torch.cat(
        features,
        dim=0
    )

    labels = torch.cat(
        labels,
        dim=0
    )

    return features, labels


def knn_predict(
    train_features,
    train_labels,
    test_features,
    k=5
):
    """
    Predict labels using cosine-similarity k-NN.
    """

    # Calculate cosine similarity
    similarities = torch.matmul(
        test_features,
        train_features.T
    )

    # Find k nearest samples
    _, indices = similarities.topk(
        k=k,
        dim=1
    )

    nearest_labels = train_labels[
        indices
    ]

    predictions = []

    for labels in nearest_labels:

        values, counts = torch.unique(
            labels,
            return_counts=True
        )

        prediction = values[
            counts.argmax()
        ]

        predictions.append(
            prediction
        )

    return torch.stack(
        predictions
    )


def knn_accuracy(
    predictions,
    labels
):
    """
    Calculate classification accuracy.
    """

    correct = (
        predictions == labels
    ).sum().item()

    total = labels.size(0)

    accuracy = correct / total

    return accuracy
