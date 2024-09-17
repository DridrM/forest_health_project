import json
import pickle
from csv import DictReader
from enum import StrEnum
from io import BytesIO, StringIO
from typing import Dict, List, Optional

from google.cloud import storage
from google.oauth2 import service_account
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from shapely.geometry import Polygon

from fhealth.params import (
    GCP_BUCKET_NAME,
    GCP_PROJECT_ID,
    GOOGLE_APPLICATION_CREDENTIALS,
)


class DataStatus(StrEnum):
    train = "train"
    valid = "valid"
    test = "test"


class GCPDataHandler(BaseModel):
    """
    A handler for loading and managing data from a GCP bucket.
    This object is responsible for connecting to a GCP bucket using credentials,
    downloading various types of data (such as images, pickles), and handling data status.

    Fields:
    ----------
    gcp_credentials: str
        Path to the Google Cloud credentials JSON file (set via environment variable).
    project_id: str
        The GCP project ID (set via environment variable).
    bucket_name: str
        The name of the GCP bucket where data is stored (set via environment variable).
    client: Optional[storage.Client]
        A GCP Storage Client instance used to interact with the GCP bucket.
    bucket: Optional[storage.Bucket]
        A GCP bucket instance.
    rgb_image: Optional[Image.Image]
        A PIL RGB image loaded from the GCP bucket.
    mask: Optional[Any]
        A pickle object representing mask data loaded from the GCP bucket.
    data_status: DataStatus
        Status of the data (train, valid, or test).
    """

    gcp_credentials: str = Field(
        GOOGLE_APPLICATION_CREDENTIALS, kw_only=True, repr=False, exclude=True
    )
    project_id: str = Field(GCP_PROJECT_ID, kw_only=True, repr=False, exclude=True)
    bucket_name: str = Field(GCP_BUCKET_NAME, kw_only=True, repr=False, exclude=True)

    client: Optional[storage.Client] = Field(
        None, init=False, description="GCP client instance"
    )
    bucket: Optional[storage.Bucket] = Field(
        None, init=False, description="GCP bucket instance"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data) -> None:
        """
        Initializes the GCPDataHandler by setting up the GCP client and bucket instance.

        It reads the GCP credentials, creates a GCP Storage Client, and initializes the bucket instance.

        :param data: Arguments for initialization, including bucket_name and project_id.
        """

        super().__init__(**data)

        # Load google credentials
        with open(self.gcp_credentials) as source:
            info = json.load(source)

        storage_credentials = service_account.Credentials.from_service_account_info(
            info
        )

        # Create the bucket instance
        self.client = storage.Client(
            project=self.project_id, credentials=storage_credentials
        )
        self.bucket = self.client.bucket(self.bucket_name)

    def download_blob(self, blob_name: str) -> bytes:
        """
        Downloads the blob from the GCP bucket as bytes.

        :param blob_name: Name of the blob in the bucket.
        :return: Blob data as bytes.
        """
        blob = self.bucket.blob(blob_name)
        blob_data = blob.download_as_bytes()
        return blob_data


class GCPImageHandler(BaseModel, GCPDataHandler):
    """
    A handler class that extends the GCPDataHandler to manage RGB images and masks
    retrieved from Google Cloud Storage (GCS) blobs. It supports loading and managing
    image data and mask data (as Shapely polygons), and allows setting the data status
    (train, valid, or test) for the object.

    Fields:
    ----------
        rgb_image (Optional[Image.Image]): RGB PIL image object loaded from GCP.
        mask (Optional[list[Polygon]]): List of Shapely polygons representing the mask data.
        data_status (DataStatus): Data status representing the phase (train, valid, or test).
    """

    rgb_image: Optional[Image.Image] = Field(
        None, init=False, description="RGB PIL image object"
    )
    mask: Optional[list[Polygon]] = Field(
        None, init=False, description="Pickle object of the mask data"
    )
    data_status: DataStatus = Field(
        None, init=False, description="Train, valid or test data"
    )

    def load_rgb_image(self, blob_name: str) -> None:
        """
        Loads an RGB image from the GCP bucket as a PIL Image and stores it in the container.

        Args:
            blob_name (str): The name of the blob containing the image data.
        """
        image_data = self.download_blob(blob_name)
        rgb_image = Image.open(BytesIO(image_data))
        self.rgb_image = rgb_image

    def load_mask(self, blob_name: str) -> None:
        """
        Loads a pickle object representing mask data from the GCP bucket, and converts it
        into a list of Shapely polygons.

        Args:
            blob_name (str): The name of the blob containing the pickle data.
        """
        pickle_data = self.download_blob(blob_name)
        polygons = pickle.loads(pickle_data)
        self.mask = list(polygons.geoms)

    def set_data_status(self, status: DataStatus) -> None:
        """
        Set the current data status for the handler (either 'train', 'valid', or 'test').

        Args:
            status (DataStatus): The new data status to assign.
        """
        self.data_status = status


class GCPCsvHandler(BaseModel, GCPDataHandler):
    """
    A handler class that extends the GCPDataHandler to manage CSV files retrieved from
    Google Cloud Storage (GCS). It supports loading and storing CSV data as a list of dictionaries.

    Fields:
    ----------
        csv_data (Optional[List[Dict[str, str]]]): List of dictionaries where each dictionary
        represents a row from the CSV, with keys as column names.
    """

    csv_data: Optional[List[Dict[str, str]]] = Field(
        None, init=False, description="List of dictionaries representing CSV data"
    )

    def load_csv(self, blob_name: str) -> None:
        """
        Loads a CSV file from the GCP bucket and stores it as a list of dictionaries.

        Args:
            blob_name (str): The name of the blob containing the CSV data.
        """
        csv_data_bytes = self.download_blob(blob_name)
        csv_text = csv_data_bytes.decode("utf-8")
        csv_reader = DictReader(StringIO(csv_text))

        self.csv_data = [row for row in csv_reader]
