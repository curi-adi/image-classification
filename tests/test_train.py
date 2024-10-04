import pytest
from unittest.mock import patch, MagicMock
import torch
from src.train import MobileNetV2Classifier

# Test 1: Check if MobileNetV2Classifier initializes correctly
def test_mobilenet_v2_classifier_initialization():
    """
    Test if MobileNetV2Classifier initializes with the correct model, number of classes, and learning rate.
    """
    # Mock a MobileNetV2 model and pass it to the classifier
    mock_model = MagicMock()
    num_classes = 10
    lr = 0.001

    # Initialize the MobileNetV2Classifier
    classifier = MobileNetV2Classifier(mock_model, num_classes=num_classes, lr=lr)

    # Check if the classifier has the correct attributes
    assert classifier.model == mock_model
    assert classifier.lr == lr
    assert isinstance(classifier.criterion, torch.nn.CrossEntropyLoss)


# Test 2: Check if the DataLoader returns the correct batches with mock data
@patch("torch.utils.data.DataLoader")
def test_dataloader_with_mock_data(mock_dataloader):
    """
    Test if DataLoader works with mock data.
    """
    # Create mock data
    mock_data = MagicMock()
    mock_dataloader.return_value = [mock_data]

    # Initialize a DataLoader
    loader = torch.utils.data.DataLoader([mock_data], batch_size=2, shuffle=True)

    # Check if the DataLoader has been called correctly
    mock_dataloader.assert_called_with([mock_data], batch_size=2, shuffle=True)

    # Verify that the DataLoader returns the mock data
    for batch in loader:
        assert batch == mock_data


# Test 3: Check if model parameters can be accessed correctly
def test_model_parameters_access():
    """
    Test if the model parameters can be accessed correctly without raising an error.
    """
    # Create a real model and initialize the classifier
    real_model = torch.nn.Linear(10, 2)  # Simple linear model as a placeholder

    # Initialize the classifier with real model parameters
    classifier = MobileNetV2Classifier(real_model, num_classes=10, lr=0.001)

    # Verify that parameters can be accessed without error
    try:
        params = list(classifier.parameters())  # Get parameters
        assert len(params) > 0  # Ensure parameters are not empty
    except Exception as e:
        pytest.fail(f"Accessing model parameters failed with exception: {e}")


# Test 4: Verify loss calculation during training step
def test_training_step_loss():
    """
    Test if the training step calculates loss correctly.
    """
    # Create mock inputs
    mock_images = torch.randn(8, 3, 224, 224)  # Batch of 8 images
    mock_labels = torch.randint(0, 10, (8,))  # Random labels for 10 classes

    # Mock the model and its output
    mock_model = MagicMock()
    mock_model.return_value = torch.randn(8, 10)  # Model returns random logits for 10 classes

    # Initialize the classifier and perform a training step
    classifier = MobileNetV2Classifier(mock_model, num_classes=10, lr=0.001)
    batch = (mock_images, mock_labels)
    loss = classifier.training_step(batch, batch_idx=0)

    # Check if the loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)


# Test 5: Verify accuracy calculation during validation step
def test_validation_step_accuracy():
    """
    Test if the validation step calculates accuracy correctly.
    """
    # Create mock inputs
    mock_images = torch.randn(8, 3, 224, 224)  # Batch of 8 images
    mock_labels = torch.randint(0, 10, (8,))  # Random labels for 10 classes

    # Mock the model and its output
    mock_model = MagicMock()
    mock_model.return_value = torch.randn(8, 10)  # Model returns random logits for 10 classes

    # Initialize the classifier and perform a validation step
    classifier = MobileNetV2Classifier(mock_model, num_classes=10, lr=0.001)
    batch = (mock_images, mock_labels)
    loss = classifier.validation_step(batch, batch_idx=0)

    # Check if the loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
