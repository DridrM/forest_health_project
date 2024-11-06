import torch.nn as nn
import torch.nn.functional as F


class CNNBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        """ """
        super().__init__()

        self.seq_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        """ """
        x = F.relu(self.seq_block(x))

        return x
