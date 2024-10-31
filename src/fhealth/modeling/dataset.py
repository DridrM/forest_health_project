import pathlib
from typing import Any

import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

from fhealth.dataset.load_data import load_data_in_cache
from fhealth.params import (
    LOCAL_BLENDED_RGB_IMAGE_NAME,
    LOCAL_DATA_FOLDERS,
    LOCAL_MASK_IMAGE_NAME,
    LOCAL_METADATA_PATH,
)


class ForestHealthDataset(Dataset):

    def __init__(
        self,
        data_status: str,
        download: bool,
        store_resolution: int | None = None,
        transform: Any | None = None,
        target_transform: Any | None = None,
    ):
        """"""
        self.data_status = data_status
        self.data_path = pathlib.Path(LOCAL_DATA_FOLDERS[self.data_status])
        self.metadata = pd.read_csv(LOCAL_METADATA_PATH).query(
            f"data_status == {self.data_status}"
        )
        self.transform = transform
        self.target_transform = target_transform

        if download:
            load_data_in_cache(store_resolution)

    def __len__(self):
        """"""
        return len(self.metadata)

    def __getitem__(self, index):
        """"""
        # Extract local image and label folder name
        image_label_local_folder = self.metadata.iloc[index, 5]

        # Get image
        image_path = (
            self.data_path / image_label_local_folder / LOCAL_BLENDED_RGB_IMAGE_NAME
        )
        image = read_image(image_path)

        # Get labels (mask image)
        labels_path = self.data_path / image_label_local_folder / LOCAL_MASK_IMAGE_NAME
        mask_image = read_image(labels_path)

        # Transform the image if transform is not None
        if self.transform:
            image = self.transform(image)

        # Transform the target if target_transform is not None
        if self.target_transform:
            mask_image = self.target_transform(mask_image)

        return image, mask_image
