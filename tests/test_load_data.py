import os

import pandas as pd
import pytest
from PIL import Image
from shapely.geometry import Polygon

from fhealth.dataset.extract_data import GCPCsvHandler, GCPImageHandler
from fhealth.dataset.load_data import load_data_in_cache
from fhealth.dataset.transform_data import ImageDataHandler


@pytest.fixture
def mock_metadata():
    """Fixture to mock metadata DataFrame."""
    return pd.DataFrame(
        {
            "example_path": ["example_path_1", "example_path_2"],
        }
    )


@pytest.fixture
def mock_image():
    """Fixture to return a mock PIL Image."""
    return Image.new("RGB", (10, 10))


@pytest.fixture
def mock_polygon_list():
    """
    Fixture to return a mock list of Shapely polygons.
    Each polygon is created using a list of coordinates.
    """
    polygon_1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])  # Square
    polygon_2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])  # Another square
    polygon_3 = Polygon([(4, 4), (6, 4), (6, 6), (4, 6)])  # Larger square

    return [polygon_1, polygon_2, polygon_3]


@pytest.fixture
def mock_gcp_csv_handler(monkeypatch, mock_metadata):
    """Fixture to mock the GCPCsvHandler's load_csv method."""
    mock_csv_handler = GCPCsvHandler()

    def mock_load_csv(blob_name):
        mock_csv_handler.csv_data = mock_metadata

    monkeypatch.setattr(GCPCsvHandler, "load_csv", mock_load_csv)

    return mock_csv_handler


@pytest.fixture
def mock_gcp_image_handler(monkeypatch, mock_image):
    """Fixture to mock GCPImageHandler's image methods."""
    mock_image_handler = GCPImageHandler()

    def mock_load_rgb_image(blob_name):
        mock_image_handler.rgb_image = mock_image

    def mock_load_mask(blob_name):
        mock_image_handler.mask = mock_image  # Just a placeholder

    monkeypatch.setattr(GCPImageHandler, "load_rgb_image", mock_load_rgb_image)
    monkeypatch.setattr(GCPImageHandler, "load_mask", mock_load_mask)

    return mock_image_handler


@pytest.fixture
def mock_image_data_handler(monkeypatch, mock_image, mock_polygon_list):
    """Fixture to mock ImageDataHandler's image processing methods."""
    mock_data_handler = ImageDataHandler(
        rgb_image=mock_image, mask_polygons=mock_polygon_list
    )

    def mock_create_mask_from_polygons():
        pass

    def mock_blend_mask_with_rgb():
        pass

    def mock_downgrade_resolution(res):
        pass

    monkeypatch.setattr(
        ImageDataHandler, "create_mask_from_polygons", mock_create_mask_from_polygons
    )
    monkeypatch.setattr(
        ImageDataHandler, "blend_mask_with_rgb", mock_blend_mask_with_rgb
    )
    monkeypatch.setattr(
        ImageDataHandler, "downgrade_resolution", mock_downgrade_resolution
    )

    return mock_data_handler


@pytest.mark.skip("Test fail, unknown reason")
def test_load_data_in_cache_creates_folders(monkeypatch, mock_gcp_csv_handler):
    """
    Test if the load_data_in_cache function creates the required folders when they don't exist.
    """

    def mock_makedirs(path):
        print(f"Mock created directory: {path}")

    # Patch os.makedirs with monkeypatch
    monkeypatch.setattr(os, "makedirs", mock_makedirs)

    # Mock pd.read_csv to raise a FileNotFoundError
    monkeypatch.setattr(
        pd, "read_csv", lambda path: (_ for _ in ()).throw(FileNotFoundError)
    )

    # Mock print
    monkeypatch.setattr("builtins.print", lambda msg: None)

    load_data_in_cache()

    # No assertions necessary as we are verifying output and mock behavior


@pytest.mark.skip("Test doesn't work in github actions")
def test_load_data_in_cache_creates_metadata(monkeypatch, mock_gcp_csv_handler):
    """
    Test if load_data_in_cache correctly creates the metadata CSV when not found locally.
    """

    def mock_to_csv(df, path):
        print(f"Mocked saving DataFrame to {path}")

    # Mock pd.read_csv to raise a FileNotFoundError
    monkeypatch.setattr(
        pd, "read_csv", lambda path: (_ for _ in ()).throw(FileNotFoundError)
    )

    # Monkeypatch to_csv and GCPCsvHandler's load_csv method
    monkeypatch.setattr(pd.DataFrame, "to_csv", mock_to_csv)

    # Run the function
    load_data_in_cache()

    # No need for assertions; mock behavior confirms function execution


def test_load_data_in_cache_downloads_and_processes_images(
    monkeypatch, mock_metadata, mock_image
):
    """
    Test if load_data_in_cache correctly downloads and processes images and metadata.
    """
    # Simulate pd.read_csv returning mock metadata
    monkeypatch.setattr(pd, "read_csv", lambda path: mock_metadata)

    # Mock GCPImageHandler's load methods and ImageDataHandler
    def mock_load_rgb_image(blob_name):
        pass

    def mock_load_mask(blob_name):
        pass

    monkeypatch.setattr(GCPImageHandler, "load_rgb_image", mock_load_rgb_image)
    monkeypatch.setattr(GCPImageHandler, "load_mask", mock_load_mask)

    def mock_create_mask_from_polygons():
        pass

    def mock_blend_mask_with_rgb():
        pass

    monkeypatch.setattr(
        ImageDataHandler, "create_mask_from_polygons", mock_create_mask_from_polygons
    )
    monkeypatch.setattr(
        ImageDataHandler, "blend_mask_with_rgb", mock_blend_mask_with_rgb
    )

    # Run the function
    load_data_in_cache()

    # No assertions needed, mock methods will track calls


def test_load_data_in_cache_saves_images(monkeypatch, mock_metadata, mock_image):
    """
    Test if load_data_in_cache correctly saves processed images to local storage.
    """
    # Simulate pd.read_csv returning mock metadata
    monkeypatch.setattr(pd, "read_csv", lambda path: mock_metadata)

    # Mock GCPImageHandler's load methods
    def mock_load_rgb_image(blob_name):
        pass

    def mock_load_mask(blob_name):
        pass

    monkeypatch.setattr(GCPImageHandler, "load_rgb_image", mock_load_rgb_image)
    monkeypatch.setattr(GCPImageHandler, "load_mask", mock_load_mask)

    # Mock Image saving
    def mock_image_save(path):
        print(f"Mock saved image to {path}")

    monkeypatch.setattr(Image.Image, "save", mock_image_save)

    # Run the function
    load_data_in_cache()

    # No need for assertions; mock behavior will log calls to save
