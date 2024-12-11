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
        - downhill (int, default 4): The number of couples CNNBlocks / MaxPool2d layers to chain together.
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
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
    """
    The purpose of the Decoder is to match the extracted features received
    from the Encoder to pixel-by-pixel labelised masks. To do so the
    Decoder is (almost) the mirror image of the Encoder. It has several
    stages that take into input the previous Decoder stage output and the
    corresponding Encoder stage's output. At the end, an exit layer map the
    number of channels of the training image with the number of channels of
    the corresponding mask.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exit_channels: int,
        padding: int,
        uphill_div: int = 2,
        uphill: int = 4,
    ) -> None:
        """
        Chain together multiple couples ConvTranspose2D / CNNBlocks together.
        The ConvTranspose2D layer upsample a tensor with `in_channels` channels
        to a tensor with `out_channels` channels (invers operation of Conv2D).
        Here, at the opposite of the Encoder, we divide both `in_channels` and
        `out_channels` by the same integer (by default 2).

        Args:
        - in_channels (int): The number of input channels of the first CNNBlocks object.
        - out_channels (int): The number of output channels of the first CNNBlocks object.
        - exit_channels (int): The number of channels of the mask.
        - padding (int): The padding parameter (offset pixel starting point of kernels) for CNNBlocks.

        Params:
        - uphill_div (int, default 2): The divider from the number of input / output channels from one step to another.
        - uphill (int, default 4): The number of couples CNNBlocks / convTranspose2D layers to chain together.
        """
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

    def forward(self, x: torch.Tensor, bypass: list[torch.Tensor]) -> torch.Tensor:
        """
        If the block is of type CNNBlocks, we take the last element of the
        bypass list from the Encoder, we apply a crop and a center transformation,
        and we stack the bypass element to the input destinated to the CNNBlocks.
        We then iteratively compute either the output of CNNBlocks either the
        ConvTranspose2D's one.

        Args:
        - x (torch.Tensor): A torch tensor representing a multi-channels encoded image.
        - bypass (list of torch Tensors): The list of non-Maxpooled intermediate results from the Encoder

        Returns:
        - x (torch.Tensor): A tensor representing a predicted mask.
        """
        bypass.pop(-1)  # We don't use the last element from the tip of the 'U'

        for blocks in self.decoder_blocks:
            # Center, crop and stack the results of corresponding
            # level in the Encoder to the input of the curent couple CNNBlocks / ConvTranspose2D
            if isinstance(blocks, CNNBlocks):
                bypass[-1] = center_crop(bypass[-1], output_size=x.shape[2])
                x = torch.cat([x, bypass.pop(-1)], dim=1)

            x = blocks(x)

        return x


class Unet(nn.Module):
    """
    Pack together the Encoder and the Decoder part of the U-net.
    """

    def __init__(
        self,
        in_channels: int,
        first_out_channels: int,
        exit_channels: int,
        downhill: int,
        interblock_mult: int = 2,
        padding: int = 0,
    ) -> None:
        """
        Instanciate the Encoder and the Decoder.
        For the Decoder, the `in_channels` and the `out_channels` are calculated
        using the depth of the U-net (`downhill` argument) and the multiplicative
        factor for the number of channels at each stage of the U-net (`interblock_mult`).

        Args:
        - in_channels (int): The number of channels of the input image.
        - first_out_channels (int): The number of output channels of the first CNNBlock.
        - exit_channels (int): The number of channels of the mask.
        - downhill (int): The number of stages of the U-net.

        Params:
        - interblock_mult (int): The multiplicator integer of output channels between stages.
        - padding (int): Number of offset pixel in each dimension the kernel has to start on at beginning of each conv.
        """
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=first_out_channels,
            downhill=downhill,
            padding=padding,
            downhill_mult=interblock_mult,
        )
        self.decoder = Decoder(
            in_channels=first_out_channels * (interblock_mult**downhill),
            out_channels=first_out_channels * (interblock_mult ** (downhill - 1)),
            exit_channels=exit_channels,
            padding=padding,
            uphill=downhill,
            uphill_div=interblock_mult,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the Encoder and the Decoder part of the U-net.

        Args:
        - x (torch.Tensor): A torch tensor representing a multi-channels image.

        Returns: A predicted mask (torch.Tensor).
        """
        x, bypass = self.encoder(x)
        x = self.decoder(x, bypass)

        return x
