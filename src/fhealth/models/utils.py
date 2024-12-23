import torch.nn as nn

# Define a type alias for torch models
type NNModel = nn.Module


def count_parameters(model: NNModel) -> int:
    """
    Count the number of trainable parameters inside a
    torch neural network model.

    Args:
    - model (NNModel): A torch neural network model.

    Returns: The number (int) of trainable parameters inside the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
