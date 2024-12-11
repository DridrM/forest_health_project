import pathlib

import pandas as pd
import pytest
import torch

# import torchvision.io
from torchvision.transforms import Normalize

import fhealth.dataset.load_data
import fhealth.train.create_dataset
from fhealth.train.create_dataset import ForestHealthDataset


@pytest.fixture
def mock_local_root_data_path(monkeypatch):
    """
    Mock the root data path on my machine as it cannot be accessed
    by github actions when testing.
    """
    fake_local_data_folders = {
        "train": "fake/data/folder",
        "valid": "fake/data/folder",
        "test": "fake/data/folder",
    }

    monkeypatch.setattr(
        fhealth.train.create_dataset, "LOCAL_DATA_FOLDERS", fake_local_data_folders
    )


@pytest.fixture
def mock_read_csv(monkeypatch):
    """
    Create a fake metadata dataframe.
    """
    fake_metadata = pd.DataFrame(
        {
            "label": ["Timber plantation", "Other", "Grassland shrubland"],
            "example_path": [
                "4.430849118860583_96.1016343478138",
                "1.3323406178609702_109.37422873130464",
                "1.720266384577504_115.00699582064485",
            ],
            "data_status": ["train", "valid", "test"],
        }
    )

    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: fake_metadata)


@pytest.fixture
def mock_decode_image(monkeypatch):
    """
    Monkeypatch the `torchvision.io.decode_image` function.
    """
    fake_image_and_mask = torch.rand(64, 64).unsqueeze(0)

    monkeypatch.setattr(
        fhealth.train.create_dataset,
        "decode_image",
        lambda *args, **kwargs: fake_image_and_mask,
    )


@pytest.fixture
def fake_transform():
    """
    Create a Lambda object for the features and target transformation.
    """
    return Normalize((0.5,), (0.5,))


@pytest.fixture
def fake_forest_health_dataset(
    monkeypatch,
    mock_local_root_data_path,
    mock_read_csv,
    mock_decode_image,
    fake_transform,
):
    """
    Fixture that initialize a ForestHealthDataset object with mocked data_path and metadata fields.
    """
    # Set data_status to 'train'
    data_status = "train"

    # Mock the load_data_in_cache function
    monkeypatch.setattr(
        fhealth.dataset.load_data, "load_data_in_cache", lambda *args: None
    )

    # Initialize the ForestHealthDataset object
    dataset = ForestHealthDataset(
        data_status=data_status,
        download=True,
        store_resolution=(64, 64),
        transform=fake_transform,
        target_transform=fake_transform,
    )

    return dataset


def test_init(fake_forest_health_dataset):
    """
    Test the initialisation of the `ForestHealthDataset` object.
    """
    assert fake_forest_health_dataset.data_status == "train"
    assert fake_forest_health_dataset.data_path == pathlib.Path("fake/data/folder")
    assert isinstance(fake_forest_health_dataset.metadata, pd.DataFrame)
    assert isinstance(fake_forest_health_dataset.transform, Normalize)
    assert isinstance(fake_forest_health_dataset.target_transform, Normalize)


def test_len(fake_forest_health_dataset):
    """
    Test the `__len__` method of the `ForestHealthDataset` object.
    """
    assert len(fake_forest_health_dataset) == 1


def test_getitem(fake_forest_health_dataset):
    """
    Test the `__getitem__` method from the `ForestHealthDataset` object.
    """
    # Extract image and mask from the mock_forest_health_dataset object
    image, mask = fake_forest_health_dataset[0]

    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert image.size() == (1, 64, 64)
