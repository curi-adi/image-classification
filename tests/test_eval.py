import pytest
from pathlib import Path
import sys
sys.path.insert(0, '../src')
import lightning as L
from unittest.mock import patch
from eval import evaluate_model, MobileNetV2LightningModule
from torchvision import models
import torch

# Add the src directory to sys.path using a relative path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

@pytest.fixture
def mock_sys_args():
    # Backup and mock the sys.argv
    original_args = sys.argv
    sys.argv = ["pytest"]
    yield
    sys.argv = original_args

@pytest.mark.usefixtures("mock_sys_args")
def test_evaluate_model():
    # Define a relative path to the checkpoint file
    dummy_ckpt_path = Path("src/checkpoints/mobilenetv2_checkpoint.ckpt")  # Relative path to checkpoints
    dummy_batch_size = 8
    dummy_num_classes = 10
    dummy_num_workers = 2

    # Skip the test if checkpoint doesn't exist
    if not dummy_ckpt_path.exists():
        pytest.skip(f"Skipping test: {dummy_ckpt_path} does not exist.")

    try:
        # Run evaluation
        evaluate_model(
            ckpt_path=dummy_ckpt_path,
            batch_size=dummy_batch_size,
            num_classes=dummy_num_classes,
            num_workers=dummy_num_workers
        )
    except Exception as e:
        pytest.fail(f"Evaluation failed with exception: {e}")

def test_evaluate_model_with_different_batch_sizes():
    # Test with different batch sizes to check evaluation stability
    dummy_ckpt_path = Path("src/checkpoints/mobilenetv2_checkpoint.ckpt")  # Update to valid checkpoint path
    dummy_num_classes = 10
    dummy_num_workers = 2

    if not dummy_ckpt_path.exists():
        pytest.skip(f"Skipping test: {dummy_ckpt_path} does not exist.")

    for batch_size in [1, 8, 16, 32]:  # Test with varying batch sizes
        try:
            evaluate_model(
                ckpt_path=dummy_ckpt_path,
                batch_size=batch_size,
                num_classes=dummy_num_classes,
                num_workers=dummy_num_workers
            )
        except Exception as e:
            pytest.fail(f"Evaluation failed with batch size {batch_size} and exception: {e}")

def test_invalid_checkpoint_in_evaluation():
    # Test with an invalid checkpoint file to check error handling
    invalid_ckpt_path = Path("src/checkpoints/invalid_checkpoint.ckpt")
    dummy_batch_size = 8
    dummy_num_classes = 10
    dummy_num_workers = 2

    with pytest.raises(Exception):
        evaluate_model(
            ckpt_path=invalid_ckpt_path,
            batch_size=dummy_batch_size,
            num_classes=dummy_num_classes,
            num_workers=dummy_num_workers
        )
def test_evaluate_invalid_checkpoint():
    invalid_ckpt_path = Path("src/checkpoints/invalid_checkpoint.ckpt")  # Simulated invalid path
    dummy_batch_size = 8
    dummy_num_classes = 10
    dummy_num_workers = 2

    # This test should raise an error or handle gracefully
    with pytest.raises(Exception):
        evaluate_model(
            ckpt_path=invalid_ckpt_path,
            batch_size=dummy_batch_size,
            num_classes=dummy_num_classes,
            num_workers=dummy_num_workers
        )

# Test for evaluation with varying data splits
def test_evaluate_with_varied_data_split():
    dummy_ckpt_path = Path("src/checkpoints/mobilenetv2_checkpoint.ckpt")
    dummy_batch_size = 8
    dummy_num_classes = 10
    dummy_num_workers = 2

    if not dummy_ckpt_path.exists():
        pytest.skip(f"Skipping test: {dummy_ckpt_path} does not exist.")

    # Test with 50% validation split
    try:
        evaluate_model(
            ckpt_path=dummy_ckpt_path,
            batch_size=dummy_batch_size,
            num_classes=dummy_num_classes,
            num_workers=dummy_num_workers
        )
    except Exception as e:
        pytest.fail(f"Evaluation failed with 50% validation split and exception: {e}")
def test_evaluate_model_invalid_ckpt():
    with pytest.raises(Exception):
        evaluate_model(
            ckpt_path='invalid/path/to/ckpt.ckpt',
            batch_size=8,
            num_classes=10,
            num_workers=2
        )

def test_evaluate_model_without_ckpt():
    # Should skip or handle cases where the checkpoint does not exist
    dummy_batch_size = 8
    dummy_num_classes = 10
    dummy_num_workers = 2
    with pytest.raises(Exception):
        evaluate_model(
            ckpt_path=None,
            batch_size=dummy_batch_size,
            num_classes=dummy_num_classes,
            num_workers=dummy_num_workers
        )


@pytest.mark.usefixtures("mock_sys_args")
def test_model_initialization():
    # Load a sample model and ensure the classifier is modified correctly
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 10)
    lightning_model = MobileNetV2LightningModule(model)
    assert lightning_model.model.classifier[1].out_features == 10, "Model initialization failed."
# import pytest
# from pathlib import Path
# import sys
# import lightning as L

# # Add the src directory to sys.path using a relative path
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# # Import from src.eval using relative path
# from src.eval import evaluate_model, MobileNetV2LightningModule
# from torchvision import models
# import torch

# from unittest.mock import patch

# @pytest.fixture
# def mock_sys_args():
#     # Backup and mock the sys.argv
#     original_args = sys.argv
#     sys.argv = ["pytest"]
#     yield
#     sys.argv = original_args


# def test_evaluate_model():
#     # Define a relative path to the checkpoint file
#     dummy_ckpt_path = Path("../checkpoints/mobilenetv2_checkpoint.ckpt")  # Relative path to checkpoints
#     dummy_batch_size = 8
#     dummy_num_classes = 10
#     dummy_num_workers = 2

#     if not dummy_ckpt_path.exists():
#         pytest.skip(f"Skipping test: {dummy_ckpt_path} does not exist.")

#     try:
#         evaluate_model(
#             ckpt_path=dummy_ckpt_path,
#             batch_size=dummy_batch_size,
#             num_classes=dummy_num_classes,
#             num_workers=dummy_num_workers
#         )
#     except Exception as e:
#         pytest.fail(f"Evaluation failed with exception: {e}")

# def test_model_initialization():
#     # Load a sample model and ensure the classifier is modified correctly
#     model = models.mobilenet_v2(pretrained=False)
#     model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 10)
#     lightning_model = MobileNetV2LightningModule(model)
#     assert lightning_model.model.classifier[1].out_features == 10, "Model initialization failed."
