import matplotlib.pyplot as plt
import torch
import torchvision


def show_batch_images(
    images: torch.Tensor, nrows: int = 8, figsize: tuple = (14, 8)
) -> None:
    """
    Display a grid of images using Matplotlib.

    This function takes a batch of images (which are single-channel grayscale images)
    and arranges them in a grid. It uses `torchvision.utils.make_grid()` to create the grid and
    `matplotlib.pyplot` to display the images.

    Args:
        images (torch.Tensor): A batch of images in the form of a PyTorch tensor.
                               The tensor should have the shape (N, 1, H, W), where
                               N is the number of images, 1 is the single channel,
                               and H and W are the height and width of the images.
        nrows (int, optional): The number of images to display per row in the grid.
                               Defaults to 8.
        figsize (tuple, optional): The size of the figure to display. The tuple
                                   represents the width and height of the figure in inches.
                                   Defaults to (14, 8).

    Returns:
        None
    """
    image_grid = torchvision.utils.make_grid(
        images, nrow=nrows
    )  # make_grid create 3 channel image by default
    plt.figure(figsize=figsize)
    plt.imshow(image_grid)
    plt.axis("off")
    plt.show()
