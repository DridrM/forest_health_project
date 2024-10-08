import os

import pandas as pd

from fhealth.dataset.extract_data import GCPCsvHandler, GCPImageHandler
from fhealth.dataset.transform_data import ImageDataHandler
from fhealth.params import (
    BLEND_GIVEN_DATA_STATUS,
    GCP_MASK_PATH,
    GCP_METADATA_PATH_AND_LABELS,
    GCP_RGB_IMAGE_PATH,
    LOCAL_BLENDED_RGB_IMAGE_NAME,
    LOCAL_EXAMPLES_PATH,
    LOCAL_MASK_IMAGE_NAME,
    LOCAL_METADATA_PATH,
    LOCAL_RGB_IMAGE_NAME,
    LOCAL_TEST_FOLDER,
    LOCAL_TRAIN_FOLDER,
    LOCAL_VALID_FOLDER,
)


def load_data_in_cache(*store_resolution: int | None) -> None:
    """
    Loads data from GCP into the local cache, including CSV metadata and image data, and processes it.

    The function first checks if the local folder structure exists; if not, it creates it. It then downloads
    CSV metadata from the Google Cloud bucket and loads the metadata into a pandas DataFrame. The function
    iterates over the DataFrame and for each row, it fetches the associated RGB image and mask from the
    Google Cloud bucket, processes the image, and stores the image data locally. Optionally, it downgrades
    the image resolution.

    Args:
        store_resolution (int | None): Optional integer(s) specifying the resolution to downgrade the images.
                                        If not provided, the original resolution is used.

    Returns:
        None: This function doesn't return any value, but it saves images and metadata to local storage.
    """
    # Create the folder structure if the data folder does not exists
    path_list = [LOCAL_TRAIN_FOLDER, LOCAL_VALID_FOLDER, LOCAL_TEST_FOLDER]
    for path in path_list:
        try:
            print(f"Creating the {path} folder...")
            os.makedirs(path)

        except OSError:
            print(f"The {path} folder already exists.")
            return

    # Download the csv metadata files as one df if it doesn't already exists at local root data path
    try:
        metadata_df = pd.read_csv(LOCAL_METADATA_PATH)

    except FileNotFoundError:
        print(f"Creating the {LOCAL_METADATA_PATH} file...")

        csv_metadata = GCPCsvHandler()
        csv_metadata_gcp_path_and_labels = GCP_METADATA_PATH_AND_LABELS
        csv_metadata_list = []

        for label, csv_blob_name in csv_metadata_gcp_path_and_labels.items():
            csv_metadata.load_csv(csv_blob_name)
            csv_metadata.csv_data["data_status"] = label  # Add data_status column
            csv_metadata_list.append(csv_metadata.csv_data)

        metadata_df = pd.concat(csv_metadata_list)
        metadata_df.to_csv(LOCAL_METADATA_PATH)

    # Iterate over the metadata df:
    for _, row in metadata_df.iterrows():
        # Extract data status and GCP image path from row
        try:
            example_path, data_status = row["example_path"], row["data_status"]

        except KeyError:
            print(
                f"Check the columns names inside the {LOCAL_METADATA_PATH} file and modify the {__file__} script accordingly."
            )
            return

        # Extract image data
        image = GCPImageHandler()

        rgb_image_blob_path = f"{example_path}/{GCP_RGB_IMAGE_PATH}"
        image.load_rgb_image(rgb_image_blob_path)

        mask_blob_path = f"{example_path}/{GCP_MASK_PATH}"
        image.load_mask(mask_blob_path)

        # Transform the image data
        transformed_image = ImageDataHandler(
            rgb_image=image.rgb_image, mask_polygons=image.mask
        )
        transformed_image.create_mask_from_polygons()

        if BLEND_GIVEN_DATA_STATUS.get(data_status, True):
            transformed_image.blend_mask_with_rgb()  # Blend the rgb image with mask given the data status

        if store_resolution:
            transformed_image.downgrade_resolution(store_resolution)

        # Store the image data
        image_name = example_path.split("/")[-1]
        image_local_path = f"{LOCAL_EXAMPLES_PATH}/{data_status}/{image_name}"

        try:
            os.makedirs(image_local_path)

            if transformed_image.blended_image:
                blended_rgb_image_local_path = (
                    f"{image_local_path}/{LOCAL_BLENDED_RGB_IMAGE_NAME}"
                )
                transformed_image.blended_image.save(
                    blended_rgb_image_local_path
                )  # Save the blended rgb image if the data status allow it

            else:
                rgb_image_local_path = f"{image_local_path}/{LOCAL_RGB_IMAGE_NAME}"
                transformed_image.rgb_image.save(
                    rgb_image_local_path
                )  # Else save the rgb image

            mask_image_local_path = f"{image_local_path}/{LOCAL_MASK_IMAGE_NAME}"
            transformed_image.mask_image.save(mask_image_local_path)

        except OSError as e:
            print(e, f"The image {image_local_path} already exists.")
