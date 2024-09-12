import json
from enum import Enum

from google.cloud import storage
from google.oauth2 import service_account

from io import BytesIO
from PIL import Image
import pickle

from pydantic import BaseModel, Field
from typing import Optional, Any

from fhealth.params import GOOGLE_APPLICATION_CREDENTIALS, GCP_PROJECT_ID, GCP_BUCKET_NAME


class DataStatus(str, Enum):
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

    gcp_credentials: str = Field(GOOGLE_APPLICATION_CREDENTIALS, repr=False, exclude=True)
    project_id: str = Field(GCP_PROJECT_ID, repr=False, exclude=True)
    bucket_name: str = Field(GCP_BUCKET_NAME, repr=False, exclude=True)

    client: Optional[storage.Client] = Field(None, description="GCP client instance")
    bucket: Optional[storage.Bucket] = Field(None, description="GCP bucket instance")

    rgb_image: Optional[Image.Image] = Field(None, kw_only=True, description="RGB PIL image object")
    mask: Optional[Any] = Field(None, kw_only=True, description="Pickle object of the mask data")
    data_status: DataStatus = Field(None, kw_only=True, description="Train, valid or test data")

    class Config:
        arbitrary_types_allowed = True

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

        storage_credentials = service_account.Credentials.from_service_account_info(info)

        # Create the bucket instance
        self.client = storage.Client(project=self.project_id, credentials=storage_credentials)
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

    def load_rgb_image(self, blob_name: str) -> None:
        """
        Loads an image from the GCP blob and stores it in the container.

        :param blob_name: Name of the image blob.
        """
        image_data = self.download_blob(blob_name)
        rgb_image = Image.open(BytesIO(image_data))
        self.rgb_image = rgb_image

    def load_mask(self, blob_name: str) -> None:
        """
        Loads a pickle object representing mask data from the GCP blob and stores it in the container.

        :param blob_name: Name of the pickle blob.
        """
        pickle_data = self.download_blob(blob_name)
        polygons = pickle.loads(pickle_data)
        self.mask = list(polygons.geoms)

    def set_data_status(self, status: DataStatus) -> None:
        """
        Set the data status for the handler (train, valid, or test).

        :param status: The new status to assign to the data.
        """
        self.data_status = status
