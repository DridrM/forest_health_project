import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """"""

    def __init__(self, power: int) -> None:
        """"""
        super().__init__()

        self.power = power

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """"""
        # Prepare outputs and labels
        # Keep this section for later if necessary

        # Compute the numerator and denominator of dice loss
        numerator = 2 * (outputs * labels).sum(dim=(2, 3))
        denominator = outputs.pow(self.power).sum(dim=(2, 3)) + labels.sum(dim=(2, 3))

        return 1 - 0.5 * (numerator / denominator).mean()
