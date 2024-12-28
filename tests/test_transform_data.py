import random

import pytest
from PIL import Image
from shapely.geometry import Polygon

from fhealth.dataset.transform_data import ImageDataHandler


@pytest.fixture
def sample_rgb_image():
    """Fixture for creating a sample RGB image."""
    img = Image.new("RGB", (100, 100), color="red")
    return img


@pytest.fixture
def sample_polygons():
    """Fixture for creating sample shapely polygons."""

    # Function to generate random coordinates
    def generate_random_coordinates(num_points=5, x_range=(0, 100), y_range=(0, 100)):
        return [
            (random.uniform(*x_range), random.uniform(*y_range))
            for _ in range(num_points)
        ]

    # Generate random coordinates for the polygons
    random_coords = generate_random_coordinates()

    # Create a list of fake polygons
    fake_polygons = [Polygon(random_coords) for _ in range(random.randint(2, 10))]

    return fake_polygons


@pytest.fixture
def handler_with_data(sample_rgb_image, sample_polygons):
    """Fixture to initialize the ImageDataHandler with test data."""
    return ImageDataHandler(rgb_image=sample_rgb_image, mask_polygons=sample_polygons)


def test_create_mask_from_polygons(handler_with_data):
    """Test the creation of a mask image from polygons."""
    handler_with_data.create_mask_from_polygons()

    # Check if the mask image is created and its size matches the original image
    assert handler_with_data.mask_image is not None
    assert handler_with_data.mask_image.size == handler_with_data.rgb_image.size

    # Ensure the mask image is in grayscale mode
    assert handler_with_data.mask_image.mode == "L"


def test_downgrade_resolution(handler_with_data):
    """Test downgrading the resolution of the mask and blended image."""
    handler_with_data.create_mask_from_polygons()

    # Downgrade resolution to 50x50
    handler_with_data.downgrade_resolution(50)

    # Check if the mask and blended images are resized to 50x50
    assert handler_with_data.mask_image.size == (50, 50)
    assert handler_with_data.rgb_image.size == (50, 50)


def test_create_mask_without_polygons_raises_error(sample_rgb_image):
    """Test that creating a mask without polygons raises an error."""
    handler = ImageDataHandler(
        rgb_image=sample_rgb_image, mask_polygons=[], blending_alpha=0.5
    )

    with pytest.raises(ValueError, match="No polygons provided to create the mask."):
        handler.create_mask_from_polygons()


def test_downgrade_resolution_without_images_raises_error(handler_with_data):
    """Test that downgrading resolution without mask image raises an error."""
    # Attempt to downgrade resolution without mask and blended images
    with pytest.raises(
        ValueError, match="Mask image not available for resolution downgrade."
    ):
        handler_with_data.downgrade_resolution(50)
