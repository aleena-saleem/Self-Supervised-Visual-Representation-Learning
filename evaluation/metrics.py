
import torch


def accuracy(
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

    if total == 0:
        return 0.0

    return correct / total


def top_k_accuracy(
    logits,
    labels,
    k=5
):
    """
    Calculate top-k classification accuracy.

    logits:
        Model predictions of shape [batch, classes]

    labels:
        Ground-truth labels of shape [batch]
    """

    _, top_k_predictions = logits.topk(
        k,
        dim=1
    )

    correct = (
        top_k_predictions == labels.unsqueeze(1)
    ).any(dim=1)

    return correct.float().mean().item()
