import pathlib
import random
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image

from fhealth.dataset.load_data import load_data_in_cache
from fhealth.params import (
    BLEND_GIVEN_DATA_STATUS,
    DEFAULT_RGB_BLENDING_RATIO,
    LOCAL_DATA_FOLDERS,
    LOCAL_MASK_IMAGE_NAME,
    LOCAL_METADATA_PATH,
    LOCAL_RGB_IMAGE_NAME,
)


def blend_image_with_mask(
    rgb_image: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = DEFAULT_RGB_BLENDING_RATIO,
) -> torch.Tensor:
    """
    Blend a given RGB training image with its corresponding mask.

    Args:
    - rgb_image (torch.Tensor): A training RGB image
    - mask (torch.Tensor): A mask corresponding to a segmented region on rgb image.

    Params:
    - alpha (float): The blending ratio between rgb image and its mask.

    Returns:
    - torch.Tensor: A RGB image blended with its mask
    """
    if not (0 < alpha <= 1):
        raise ValueError("Blending alpha must be between 0 excluded and 1.")

    return alpha * mask + (1 - alpha) * rgb_image


class ForestHealthDataset(Dataset):
    """
    An object representing one of the three splits ('train', 'valid', 'test') of the Forest Health Dataset.
    This object allow the `torch.utils.data.Dataloader` to load randomly datas into a training loop.
    """

    def __init__(
        self,
        data_status: str,
        download: bool,
        prop_blend_train: float = 0.0,
        store_resolution: tuple | None = None,
        transform: Any | None = None,
        target_transform: Any | None = None,
    ) -> None:
        """
        Read the metadata.csv file to acquire the local image and mask folder path,
        Set the transformation functions for the image and the mask and download
        the full dataset if the download argument is set to `True`.

        Args:
            datastatus (str): The data status among 'train', 'valid' and 'test'.
            download (bool): Wether to download the full dataset from cloud provider.
            store_resolution (tuple | None): The dimensions (height, lenght)of the example image and the mask
                                             if you choose to download the full dataset.
            transform (Any | None): The runtime transformation you want to apply to the features (images)
            target_transform (Any | None): The runtime transformation you want to apply to the targets (masks)

        Returns:
            None
        """
        # Download the full dataset if needed
        if download and store_resolution:
            load_data_in_cache(*store_resolution)

        if download and (not store_resolution):
            load_data_in_cache()

        # Read the metadata.csv file to acquire the local image and mask folder path
        self.data_status = data_status
        self.data_path = pathlib.Path(LOCAL_DATA_FOLDERS.get(self.data_status, "train"))
        self.metadata = (
            pd.read_csv(LOCAL_METADATA_PATH)
            .query(f"data_status == '{self.data_status}'")
            .reset_index()
        )

        # Proportion of training images to blend
        self.prop_blend_train = prop_blend_train

        # Transformation functions for the image and the mask
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """
        Compute the number of samples in the dataset.

        Returns:
            The number of sample data in the dataset.
        """
        return len(self.metadata)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        This method allow the `torch.utils.data.Dataloader` object to randomly image and mask into a training loop.
        The method also apply transformations to the image and the mask if the corresponding arguments were provided
        when the `ForestHealthDataset` object was initialized.

        Args:
            index (int): The index of the couple image, mask to return.

        Returns:
            A tuple image, mask
        """
        # Extract local image and label folder name
        try:
            image_label_local_folder = self.metadata.loc[index, "example_path"].split(
                "/"
            )[-1]

        except IndexError as e:
            print(
                f"Check the names of the columns of the {LOCAL_METADATA_PATH} file.", e
            )

        # Get image
        image_path = self.data_path / image_label_local_folder / LOCAL_RGB_IMAGE_NAME
        image = decode_image(image_path)

        # Get labels (mask image)
        labels_path = self.data_path / image_label_local_folder / LOCAL_MASK_IMAGE_NAME
        mask_image = decode_image(labels_path)

        # Transform the image if transform is not None
        if self.transform:
            image = self.transform(image)

        # Transform the target if target_transform is not None
        if self.target_transform:
            mask_image = self.target_transform(mask_image)

        # Random blend the TRAINING images only with masks, given a proportion of images blended
        blend_status = BLEND_GIVEN_DATA_STATUS.get(self.data_status)
        blend = random.uniform(0, 1) <= self.prop_blend_train

        if blend_status and blend:
            image = blend_image_with_mask(image, mask_image)

        return image, mask_image


# if __name__ == "__main__":
#     training_set = ForestHealthDataset(data_status="test", download=False)
#     training_loader = DataLoader(training_set, batch_size=32, shuffle=True)

#     for images, masks in training_loader:
#         print("Images and masks succesfully loaded.")
