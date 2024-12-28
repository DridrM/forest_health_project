import torch


def binary_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute the accuracy between two binary Tensors (0 or 1) :
    Number of true positives and true negatives over the total number
    of elements inside a binary (0 or 1) Tensor.

    Args:
    - outputs (torch.Tensor): A predicted binary Tensor.
    - labels (torch.Tensor): The true binary Tensor to predict.

    Returns: A float representing the Accuracy of the prediction.
    """
    # Compute true positives sum
    true_positives = torch.logical_and(outputs != 0.0, labels != 0.0).sum()

    # Compute true negatives sum
    true_negatives = torch.logical_and(outputs == 0.0, labels == 0.0).sum()

    return ((true_positives + true_negatives) / labels.numel()).item()


def binary_precision(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute the Precision between two binary Tensors (0 or 1) :
    Number of true positives over the total number of predicted
    values inside a binary (0 or 1) Tensor.

    Args:
    - outputs (torch.Tensor): A predicted binary Tensor.
    - labels (torch.Tensor): The true binary Tensor to predict.

    Returns: A float representing the Precision of the prediction.
    """
    # Compute true positives sum
    true_positives = torch.logical_and(outputs != 0.0, labels != 0.0).sum()

    # Compute the number of false positives
    false_positives = torch.logical_and(outputs != 0.0, labels == 0.0).sum()

    return (true_positives / (true_positives + false_positives)).item()


def binary_recall(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute the Recall between two binary Tensors (0 or 1) :
    Number of true positives over the total number of True
    values inside a binary (0 or 1) Tensor.

    Args:
    - outputs (torch.Tensor): A predicted binary Tensor.
    - labels (torch.Tensor): The true binary Tensor to predict.

    Returns: A float representing the Recall of the prediction.
    """
    # Compute true positives sum
    true_positives = torch.logical_and(outputs != 0.0, labels != 0.0).sum()

    # Compute the number of false negatives
    false_negatives = torch.logical_and(outputs == 0.0, labels != 0.0).sum()

    return (true_positives / (true_positives + false_negatives)).item()
