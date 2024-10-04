import pytest
from unittest.mock import MagicMock, patch
from kaggle.api.kaggle_api_extended import KaggleApi
import sys
sys.path.insert(0, '../src')
from datamodules.catdog import DogImageDataModule
import torch
from torchvision import transforms


@pytest.fixture
def sample_data_module():
    """Fixture for creating a sample instance of DogImageDataModule."""
    return DogImageDataModule(
        data_dir="data/sample_dataset",  # Mocked data directory
        batch_size=4,
        num_workers=0,
        train_val_split=0.75  # 75% train, 25% val split
    )


@patch.object(KaggleApi, "authenticate", return_value=None)  # Mock authenticate method with no arguments
@patch.object(KaggleApi, "dataset_download_files", return_value=None)  # Mock dataset download with specific arguments
def test_prepare_data(mock_download, mock_authenticate, sample_data_module):
    """Test the prepare_data method to ensure Kaggle API calls are made correctly."""
    # Call prepare_data and check if Kaggle API calls were made
    sample_data_module.prepare_data()

    # Check that authenticate was called once without arguments
    mock_authenticate.assert_called_once()

    # Check that dataset_download_files was called once with the correct arguments
    mock_download.assert_called_once_with(
        'khushikhushikhushi/dog-breed-image-dataset',
        path=sample_data_module.data_dir.parent,
        unzip=True
    )


def test_transform_properties(sample_data_module):
    """Test the transformations applied to the training and validation datasets."""
    train_transform = sample_data_module.train_transform
    val_transform = sample_data_module.val_transform

    assert isinstance(train_transform, transforms.Compose)
    assert isinstance(val_transform, transforms.Compose)

    # Check that the transform contains specific types of transformations
    assert any(isinstance(t, transforms.Resize) for t in train_transform.transforms)
    assert any(isinstance(t, transforms.RandomHorizontalFlip) for t in train_transform.transforms)
    assert any(isinstance(t, transforms.ColorJitter) for t in train_transform.transforms)


def test_dataloader_content(sample_data_module, mocker):
    """Test the content of dataloaders to ensure they return the expected shapes."""
    # Mock datasets with length and shape
    sample_data_module.train_dataset = MagicMock()
    sample_data_module.val_dataset = MagicMock()
    sample_data_module.test_dataset = MagicMock()

    # Mock __getitem__ to return a tensor of shape [3, 224, 224] (example shape for an image)
    sample_data_module.train_dataset.__getitem__.return_value = (torch.randn(3, 224, 224), 0)  # (Image, Label)
    sample_data_module.train_dataset.__len__.return_value = 100

    train_loader = sample_data_module.train_dataloader()
    for batch in train_loader:
        images, labels = batch
        assert images.shape == (sample_data_module.batch_size, 3, 224, 224)
        assert labels.shape == (sample_data_module.batch_size,)
        break  # Only test the first batch to ensure proper shape


@patch("os.makedirs", return_value=None)
@patch("builtins.open", new_callable=MagicMock)
def test_data_dir_creation(mock_open, mock_makedirs, sample_data_module):
    """Test that the data directory is created properly."""
    # Mock prepare_data to avoid actual file operations
    with patch.object(KaggleApi, "authenticate", return_value=None), \
         patch.object(KaggleApi, "dataset_download_files", return_value=None), \
         patch.object(KaggleApi, "_load_config", return_value={"username": "user", "key": "api_key"}):

        sample_data_module.prepare_data()

    # Check that os.makedirs was called to create the directory
    mock_makedirs.assert_called_once_with(sample_data_module.data_dir, exist_ok=True)


def test_train_transform_applies_correctly(sample_data_module):
    """Test that the train_transform applies correctly on a sample image."""
    sample_image = torch.randn(3, 256, 256)  # Random image of shape (C, H, W)
    transformed_image = sample_data_module.train_transform(transforms.ToPILImage()(sample_image))
    
    # Check that the transformed image has the expected shape and type
    assert isinstance(transformed_image, torch.Tensor)
    assert transformed_image.shape == (3, 224, 224)  # Check for the resize transform


def test_val_transform_applies_correctly(sample_data_module):
    """Test that the val_transform applies correctly on a sample image."""
    sample_image = torch.randn(3, 256, 256)  # Random image of shape (C, H, W)
    transformed_image = sample_data_module.val_transform(transforms.ToPILImage()(sample_image))
    
    # Check that the transformed image has the expected shape and type
    assert isinstance(transformed_image, torch.Tensor)
    assert transformed_image.shape == (3, 224, 224)  # Check for the resize transform
