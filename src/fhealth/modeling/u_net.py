import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import center_crop


class CNNBlock(nn.Module):
    """"""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ """
        x = F.relu(self.seq_block(x))

        return x


class CNNBlocks(nn.Module):
    """"""

    def __init__(
        self,
        n_conv_blocks: int,  # Transform into a list for further versions
        in_channels: int,
        out_channels: int,
        padding: int,
    ) -> None:
        """ """
        super().__init__()

        self.blocks = nn.ModuleList()

        for _ in range(n_conv_blocks):
            self.blocks.append(CNNBlock(in_channels, out_channels, padding))

            in_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ """
        for block in self.blocks:
            x = block(x)

        return x


class Encoder(nn.Module):
    """"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: int,
        downhill_mult: int = 2,
        downhill: int = 4,
    ) -> None:
        """ """
        super().__init__()

        self.encoder_blocks = nn.ModuleList()

        for _ in range(downhill):
            self.encoder_blocks += [
                CNNBlocks(2, in_channels, out_channels, padding=padding),
                nn.MaxPool2d(2, 2),
            ]

            in_channels = out_channels
            out_channels *= downhill_mult

        # Bottom encoder blocks of the network
        self.encoder_blocks.append(
            CNNBlocks(2, in_channels, out_channels, padding=padding)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ """
        bypass = []

        for blocks in self.encoder_blocks:
            x = blocks(x)

            if isinstance(blocks, CNNBlocks):
                bypass.append(
                    x
                )  # Extract the result of the CNNBlocks before the maxpooling

        return x, bypass


class Decoder(nn.Module):
    """ """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exit_channels: int,
        padding: int,
        uphill_div: int = 2,
        uphill: int = 4,
    ) -> None:
        """ """
        super().__init__()

        self.decoder_blocks = nn.ModuleList()

        for _ in range(uphill):
            self.decoder_blocks += [
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=2, stride=2
                ),  # Must be same kernel size and stride as the Max Pool layers
                CNNBlocks(2, in_channels, out_channels, padding=padding),
            ]

            in_channels //= uphill_div
            out_channels //= uphill_div

        # Last conv 2D 1x1 output layer
        self.decoder_blocks.append(
            nn.Conv2d(in_channels, exit_channels, kernel_size=1, padding=padding)
        )

    def forward(self, x: torch.Tensor, bypass: list) -> torch.Tensor:
        """"""
        bypass.pop(-1)  # We don't use the last element from the tip of the 'U'

        for blocks in self.decoder_blocks:
            x = blocks(x)

            if isinstance(blocks, CNNBlocks):
                bypass[-1] = center_crop(bypass[-1], output_size=x.shape[2])
                x = torch.cat([x, bypass.pop(-1)], dim=1)

        return x


class Unet(nn.Module):
    """"""

    def __init__(
        self,
        in_channels: int,
        first_out_channels: int,
        exit_channels: int,
        downhill: int,
        padding: int = 0,
    ) -> None:
        """"""
        super().__init__()

        self.encoder = Encoder(
            in_channels, first_out_channels, downhill=downhill, padding=padding
        )
        self.decoder = Decoder(
            first_out_channels * (2**downhill),
            first_out_channels * (2 ** (downhill - 1)),
            exit_channels=exit_channels,
            padding=padding,
            uphill=downhill,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, bypass = self.encoder(x)
        x = self.decoder(x, bypass)

        return x
