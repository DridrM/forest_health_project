import pickle
import random
from io import BytesIO
from unittest.mock import MagicMock

import pytest
from PIL import Image
from shapely.geometry import MultiPolygon, Polygon

from fhealth.dataset.extract_data import DataStatus, GCPCsvHandler, GCPImageHandler


# Fixture to initialize GCPImageHandler with mocked GCP components
@pytest.fixture
def mock_gcp_image_handler(monkeypatch):
    """
    Fixture that initializes a GCPImageHandler instance with mocked GCP components.
    """
    # Mock opening credentials file using monkeypatch
    fake_credentials_data = '{"type": "service_account"}'
    monkeypatch.setattr(
        "builtins.open", lambda *args, **kwargs: BytesIO(fake_credentials_data.encode())
    )

    # Mock GCP credentials and client
    fake_credentials = MagicMock()
    monkeypatch.setattr(
        "google.oauth2.service_account.Credentials",
        MagicMock(from_service_account_info=lambda *args, **kwargs: fake_credentials),
    )

    # Mock the GCP Client and Bucket
    mock_client_instance = MagicMock()
    mock_bucket = MagicMock()
    monkeypatch.setattr(
        "google.cloud.storage.Client", lambda *args, **kwargs: mock_client_instance
    )

    mock_client_instance.bucket.return_value = mock_bucket

    # Initialize GCPImageHandler
    handler = GCPImageHandler(
        gcp_credentials="fake_credentials.json",
        project_id="test_project",
        bucket_name="test_bucket",
    )
    handler.client = mock_client_instance
    handler.bucket = mock_bucket

    return handler


def test_init(mock_gcp_image_handler):
    """
    Test GCPImageHandler initialization with mock credentials and client.
    """
    assert mock_gcp_image_handler.client is not None
    assert mock_gcp_image_handler.bucket is not None


def test_load_rgb_image(monkeypatch, mock_gcp_image_handler):
    """
    Test the load_rgb_image method, verifying it loads an RGB image from a GCP blob.
    """
    # Create a fake image as bytes
    fake_image_data = BytesIO()
    image = Image.new("RGB", (100, 100))  # Create a dummy image
    image.save(fake_image_data, format="PNG")
    fake_image_data.seek(0)

    # Monkeypatch the download_blob method to return the fake image bytes
    monkeypatch.setattr(
        GCPImageHandler,
        "download_blob",
        lambda *args: fake_image_data.getvalue(),
    )

    # Call the method to load the image
    mock_gcp_image_handler.load_rgb_image("fake_image_blob")

    # Assert the image was loaded into the handler as a PIL image
    assert isinstance(mock_gcp_image_handler.rgb_image, Image.Image)


def test_load_mask(monkeypatch, mock_gcp_image_handler):
    """
    Test the load_pickle method, ensuring it loads a pickle object from a GCP blob.
    """

    # Function to generate random coordinates
    def generate_random_coordinates(num_points=5, x_range=(0, 100), y_range=(0, 100)):
        return [
            (random.uniform(*x_range), random.uniform(*y_range))
            for _ in range(num_points)
        ]

    # Generate random coordinates for the polygons
    random_coords = generate_random_coordinates()

    # Mock pickle object
    fake_polygons = [Polygon(random_coords) for _ in range(random.randint(2, 10))]
    fake_multi_polygon = MultiPolygon(fake_polygons)
    fake_pickle_data = pickle.dumps(fake_multi_polygon)
    monkeypatch.setattr(
        GCPImageHandler, "download_blob", lambda *args: fake_pickle_data
    )

    # Call the method to load the pickle object
    mock_gcp_image_handler.load_mask("fake_pickle_blob")

    # Assert the mask was correctly loaded into the handler
    assert isinstance(mock_gcp_image_handler.mask, list)
    assert isinstance(mock_gcp_image_handler.mask[0], Polygon)


def test_set_data_status(mock_gcp_image_handler):
    """
    Test the set_data_status method for updating the data status field.
    """
    # Set and verify data status to 'train'
    mock_gcp_image_handler.set_data_status(DataStatus.train)
    assert mock_gcp_image_handler.data_status == DataStatus.train

    # Set and verify data status to 'valid'
    mock_gcp_image_handler.set_data_status(DataStatus.valid)
    assert mock_gcp_image_handler.data_status == DataStatus.valid

    # Set and verify data status to 'test'
    mock_gcp_image_handler.set_data_status(DataStatus.test)
    assert mock_gcp_image_handler.data_status == DataStatus.test


def test_download_blob(monkeypatch, mock_gcp_image_handler):
    """
    Test the download_blob method, ensuring it correctly downloads a blob as bytes from GCP.
    """
    # Mock blob content
    fake_blob_data = b"fake_blob_data"
    mock_blob = MagicMock()
    mock_blob.download_as_bytes.return_value = fake_blob_data
    monkeypatch.setattr(
        mock_gcp_image_handler.bucket, "blob", lambda blob_name: mock_blob
    )

    # Call the download_blob method
    blob_data = mock_gcp_image_handler.download_blob("fake_blob")

    # Assert the blob was downloaded correctly
    assert blob_data == fake_blob_data
    mock_blob.download_as_bytes.assert_called_once()


def test_load_csv(monkeypatch, mock_gcp_image_handler):
    """
    Test the load_csv method of the GCPCsvHandler, ensuring it loads CSV data from a GCP blob.
    """
    # Mock CSV data
    fake_csv_data = "name,age\nAlice,30\nBob,25"

    # Monkeypatch the download_blob method to return the fake CSV data
    monkeypatch.setattr(
        GCPCsvHandler,
        "download_blob",
        lambda *args: fake_csv_data.encode("utf-8"),
    )

    # Create an instance of GCPCsvHandler and call load_csv
    csv_handler = GCPCsvHandler(
        gcp_credentials="fake_credentials.json",
        project_id="test_project",
        bucket_name="test_bucket",
    )
    csv_handler.load_csv("fake_csv_blob")

    # Assert the CSV data was loaded correctly
    assert csv_handler.csv_data is not None
    assert len(csv_handler.csv_data) == 2
    assert csv_handler.csv_data[0]["name"] == "Alice"
    assert csv_handler.csv_data[1]["name"] == "Bob"
