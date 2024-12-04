import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import center_crop


class CNNBlock(nn.Module):
    """
    CNN block with a conv 2D layer and a batch normalisation layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        """
        Initialize the conv 2D layer and the batch norm layer.

        Args:
        - in_channels (int): The number of input channels of the cond 2D layer.
        - out_channels (int): The number of output channels of the conv 2D layer.

        Params:
        - kernel_size (int): Lenght of the square side of the kernel of the conv 2D layer (square kernel).
        - stride (int): Number of pixel in one dimension the kernel has to move on at each step.
        - padding (int): Number of offset pixel in each dimension the kernel has to start on at beginning of each conv.

        Returns:
        - None
        """
        super().__init__()

        self.seq_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the conv 2D layer and the batch norm layer, wrap them up inside
        a Relu function.

        Args:
        - x (torch.Tensor): A torch tensor representing a multi-channels image.

        Returns:
        - x (torch.Tensor): A tensor representing encoded features.
        """
        x = F.relu(self.seq_block(x))

        return x


class CNNBlocks(nn.Module):
    """
    Chain multiple CNN block together. Use the previously defined CNNBlock object.
    """

    def __init__(
        self,
        n_conv_blocks: int,
        in_channels: int,
        out_channels: int,
        padding: int,
    ) -> None:
        """
        Use the nn.ModuleList to chain multiple CNNBlock together, allowing
        easy itteration through the CNN blocks.

        Args:
        - n_conv_blocks (int): The number of CNNBlock objects to chain.
        - in_channels (int): The number of input channels of the cond 2D layer inside one CNNBlock.
        - out_channels (int): The number of output channels of the conv 2D layer inside one CNNBlock.
        - padding (int): Number of offset pixel in each dimension the kernel has to start on at beginning of each conv.

        Returns:
        - None
        """
        super().__init__()

        self.blocks = nn.ModuleList()

        for _ in range(n_conv_blocks):
            self.blocks.append(CNNBlock(in_channels, out_channels, padding=padding))

            in_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the block sequentialy.

        Args:
        - x (torch.Tensor): A torch tensor representing a multi-channels image.

        Returns:
        - x (torch.Tensor): A tensor representing encoded features.
        """
        for block in self.blocks:
            x = block(x)

        return x


class Encoder(nn.Module):
    """
    The goal of the Encoder is to extract the meaninfull features from
    the original image by chaining CNNBlocks object with increasing
    depth (more output layer the deeper).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: int,
        downhill_mult: int = 2,
        downhill: int = 4,
    ) -> None:
        """
        Chain multiple couples CNNBlocks / MaxPool2d layers together.
        Set the input channels equal to the output channels of the previous CNNBlocks object.
        At each iteration to add a couple CNNBlocks / MaxPool2d, multiply the
        number of output channels by a given integer (by default 2). In other
        words, each CNNBlocks object has twice the number of output channels than
        the number of input channels.
        At the end, add a last single CNNBlocks object at the 'tip' of the U network.

        Args:
        - in_channels (int): The number of input channels of the first CNNBlocks object.
        - out_channels (int): The number of output channels of the first CNNBlocks object.
        - padding (int): The padding parameter (offset pixel starting point of kernels) for CNNBlocks.

        Params:
        - downhill_mult (int, default 2): The multiplicator from the number of input channels to the number of output ones.
        - downhill (int, default 4): the number of couples CNNBlocks / MaxPool2d layers to chain together.
        """
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
        """
        Compute the blocks sequentialy. If the block is of type CNNBlocks,
        take the output tensor and append it inside a bypass list.
        this bypass list will store intermediate computation of the Encoder
        to pass them to the corresponding stages of the Decoder.

        Args:
        - x (torch.Tensor): A torch tensor representing a multi-channels image.

        Returns:
        - x (torch.Tensor): A tensor representing encoded features.
        - bypass (list): A list of tensors representing the intermediate stages of computation.
        """
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
                ),  # Must be same kernel size and stride as the Max Pool layers inside the encoder
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
        """"""
        x, bypass = self.encoder(x)
        x = self.decoder(x, bypass)

        return x
