import json
import pickle
from io import BytesIO, StringIO
from typing import Optional

import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from shapely.geometry import MultiPolygon, Polygon

from fhealth.params import (
    GCP_BUCKET_NAME,
    GCP_PROJECT_ID,
    GOOGLE_APPLICATION_CREDENTIALS,
)


class GCPDataHandler(BaseModel):
    """
    A handler for loading and managing data from a GCP bucket.
    This object is responsible for connecting to a GCP bucket using credentials,
    downloading various types of data (such as images, pickles).

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


class GCPImageHandler(GCPDataHandler, BaseModel):
    """
    A handler class that extends the GCPDataHandler to manage RGB images and masks
    retrieved from Google Cloud Storage (GCS) blobs. It supports loading and managing
    image data and mask data (as Shapely polygons).

    Fields:
    ----------
        rgb_image (Optional[Image.Image]): RGB PIL image object loaded from GCP.
        mask (Optional[list[Polygon]]): List of Shapely polygons representing the mask data.
    """

    rgb_image: Optional[Image.Image] = Field(
        None, init=False, description="RGB PIL image object"
    )
    mask: Optional[list[Polygon]] = Field(
        None, init=False, description="Pickle object of the mask data"
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
        Loads a pickle object representing mask data from the GCP bucket and converts it
        into a list of Shapely polygons. The object can be a single Polygon or a MultiPolygon.

        Args:
            blob_name (str): The name of the blob containing the pickle data.
        """
        # Download the pickle data from the blob
        pickle_data = self.download_blob(blob_name)
        polygons = pickle.loads(pickle_data)

        # Check if the unpickled object is a Polygon or MultiPolygon
        if isinstance(polygons, Polygon):
            self.mask = [polygons]

        elif isinstance(polygons, MultiPolygon):
            self.mask = list(polygons.geoms)

        else:
            raise TypeError(
                "The unpickled object is neither a Polygon nor a MultiPolygon."
            )


class GCPCsvHandler(GCPDataHandler, BaseModel):
    """
    A handler class that extends the GCPDataHandler to manage CSV files retrieved from Google Cloud Storage (GCS).

    Fields:
    ----------
        csv_data (Optional[pd.DataFrame]): A pandas DataFrame containing the loaded CSV data.
    """

    csv_data: Optional[pd.DataFrame] = Field(
        None, init=False, description="Pandas DataFrame representing CSV data"
    )

    def load_csv(self, blob_name: str) -> None:
        """
        Loads a CSV file from the GCP bucket and stores it as a pandas DataFrame.

        Args:
            blob_name (str): The name of the blob containing the CSV data.
        """
        csv_data_bytes = self.download_blob(blob_name)
        csv_text = csv_data_bytes.decode("utf-8")

        # Load the CSV data into a pandas DataFrame
        self.csv_data = pd.read_csv(StringIO(csv_text))
