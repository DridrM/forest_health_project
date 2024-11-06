import pathlib
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image

from fhealth.dataset.load_data import load_data_in_cache
from fhealth.params import (
    BLEND_GIVEN_DATA_STATUS,
    LOCAL_BLENDED_RGB_IMAGE_NAME,
    LOCAL_DATA_FOLDERS,
    LOCAL_MASK_IMAGE_NAME,
    LOCAL_METADATA_PATH,
    LOCAL_RGB_IMAGE_NAME,
)


class ForestHealthDataset(Dataset):
    """
    An object representing one of the three splits ('train', 'valid', 'test') of the Forest Health Dataset.
    This object allow the `torch.utils.data.Dataloader` to load randomly datas into a training loop.
    """

    def __init__(
        self,
        data_status: str,
        download: bool,
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
        # Read the metadata.csv file to acquire the local image and mask folder path
        self.data_status = data_status
        self.data_path = pathlib.Path(LOCAL_DATA_FOLDERS[self.data_status])
        self.metadata = pd.read_csv(LOCAL_METADATA_PATH).query(
            f"data_status == '{self.data_status}'"
        )

        # Transformation functions for the image and the mask
        self.transform = transform
        self.target_transform = target_transform

        # Download the full dataset if needed
        if download:
            load_data_in_cache(*store_resolution)

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
            image_label_local_folder = self.metadata.loc[index, "example_path"]

        except IndexError as e:
            print(
                f"Check the names of the columns of the {LOCAL_METADATA_PATH} file.", e
            )

        # Get image, taking into account that the name change depending on if the rgb image is blended with the mask or not
        if BLEND_GIVEN_DATA_STATUS.get(self.data_status, True):
            image_path = (
                self.data_path / image_label_local_folder / LOCAL_BLENDED_RGB_IMAGE_NAME
            )

        else:
            image_path = (
                self.data_path / image_label_local_folder / LOCAL_RGB_IMAGE_NAME
            )

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

        return image, mask_image
