import pytest
import torch
from src.models.dog_classifier import DogClassifier  # Adjust the import path as necessary

# Test for optimizer configuration
def test_configure_optimizers():
    model = DogClassifier(num_classes=10)
    optimizers = model.configure_optimizers()
    assert 'optimizer' in optimizers, "Optimizer not correctly configured."
    assert 'lr_scheduler' in optimizers, "Learning rate scheduler not configured."

# Test training step
def test_training_step():
    model = DogClassifier(num_classes=10)
    # Create dummy batch data
    dummy_images = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    dummy_labels = torch.tensor([0, 1, 2, 3])  # Labels corresponding to the batch
    batch = (dummy_images, dummy_labels)
    loss = model.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor), "Training step did not return a valid loss tensor."

# Test validation step
def test_validation_step():
    model = DogClassifier(num_classes=10)
    dummy_images = torch.randn(4, 3, 224, 224)
    dummy_labels = torch.tensor([0, 1, 2, 3])
    batch = (dummy_images, dummy_labels)
    loss = model.validation_step(batch, 0)
    assert isinstance(loss, torch.Tensor), "Validation step did not return a valid loss tensor."

# Test test step
def test_test_step():
    model = DogClassifier(num_classes=10)
    dummy_images = torch.randn(4, 3, 224, 224)
    dummy_labels = torch.tensor([0, 1, 2, 3])
    batch = (dummy_images, dummy_labels)
    loss = model.test_step(batch, 0)
    assert isinstance(loss, torch.Tensor), "Test step did not return a valid loss tensor."

def test_model_initialization():
    # Test model initialization with 10 classes
    model = DogClassifier(num_classes=10)
    
    # Check that the final fully connected layer has the correct number of output features
    assert model.model.fc[1].out_features == 10, "Model should have 10 output features."

def test_model_forward_pass():
    # Test the model's forward pass with a sample input
    model = DogClassifier(num_classes=10)
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels (RGB), 224x224 image size
    output = model(dummy_input)
    
    # Check the output shape to match the number of classes
    assert output.shape == (1, 10), "Output shape should be (1, 10) for 10 classes."
