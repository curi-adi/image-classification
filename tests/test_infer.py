import pytest
from pathlib import Path
import sys

# Add the src directory to sys.path using a relative path
sys.path.insert(0, '../src')
#sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "..src"))

import lightning as L
from unittest.mock import patch
from infer import load_model, predict_single_image, load_class_names
from PIL import Image

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
def test_load_model():
    # Use relative path to checkpoint file
    dummy_ckpt_path = Path("src/checkpoints/mobilenetv2_checkpoint.ckpt")
    num_classes = 10

    # Skip the test if checkpoint doesn't exist
    if not dummy_ckpt_path.exists():
        pytest.skip(f"Skipping test: {dummy_ckpt_path} does not exist.")

    try:
        model = load_model(ckpt_path=dummy_ckpt_path, num_classes=num_classes)
    except Exception as e:
        pytest.fail(f"Model loading failed with exception: {e}")

@pytest.mark.usefixtures("mock_sys_args")
def test_load_class_names():
    # Use relative path to dataset directory
    dummy_data_dir = Path("src/data/dataset")

    # Skip the test if dataset directory doesn't exist
    if not dummy_data_dir.exists():
        pytest.skip(f"Skipping test: {dummy_data_dir} does not exist.")

    try:
        class_names = load_class_names(dummy_data_dir)
        assert len(class_names) == 10, "Class names not loaded correctly or incorrect number of classes."
    except Exception as e:
        pytest.fail(f"Failed to load class names with exception: {e}")
def test_predict_with_invalid_image():
    # Use an invalid image file to test error handling
    dummy_ckpt_path = Path("src/checkpoints/mobilenetv2_checkpoint.ckpt")
    invalid_image_path = Path("src/invalid_image.txt")  # Use an invalid image format or path
    num_classes = 10

    if not dummy_ckpt_path.exists() or not invalid_image_path.exists():
        pytest.skip(f"Skipping test: {dummy_ckpt_path} or {invalid_image_path} does not exist.")

    model = load_model(ckpt_path=dummy_ckpt_path, num_classes=num_classes)
    class_names = [f"breed_{i}" for i in range(num_classes)]

    with pytest.raises(Exception):
        predict_single_image(model, image_path=invalid_image_path, class_names=class_names)
def test_predict_corrupted_image():
    corrupted_image_path = Path("src/corrupted_image.jpg")  # Simulated corrupted image file
    dummy_ckpt_path = Path("src/checkpoints/mobilenetv2_checkpoint.ckpt")
    num_classes = 10

    if not dummy_ckpt_path.exists() or not corrupted_image_path.exists():
        pytest.skip(f"Skipping test: {dummy_ckpt_path} or {corrupted_image_path} does not exist.")

    model = load_model(ckpt_path=dummy_ckpt_path, num_classes=num_classes)
    class_names = [f"breed_{i}" for i in range(num_classes)]

    # Should raise an error due to corrupted image
    with pytest.raises(Exception):
        predict_single_image(model, image_path=corrupted_image_path, class_names=class_names)

def test_predict_invalid_image():
    dummy_ckpt_path = Path("invalid_checkpoint.ckpt")
    dummy_image_path = Path("invalid_image.jpg")
    num_classes = 10
    with pytest.raises(Exception):
        model = load_model(ckpt_path=dummy_ckpt_path, num_classes=num_classes)
        predict_single_image(model, image_path=dummy_image_path, class_names=[])

def test_load_class_names_missing_dir():
    dummy_data_dir = Path("lightning-template-hydra-master/src/datamodules")
    with pytest.raises(Exception):
        load_class_names(dummy_data_dir)


@pytest.mark.usefixtures("mock_sys_args")
def test_predict_single_image():
    # Use relative paths for checkpoint and image
    dummy_ckpt_path = Path("src/checkpoints/mobilenetv2_checkpoint.ckpt")
    dummy_image_path = Path("src/new_images/image2.jpg")
    num_classes = 10

    # Skip the test if checkpoint or image file doesn't exist
    if not dummy_ckpt_path.exists() or not dummy_image_path.exists():
        pytest.skip(f"Skipping test: {dummy_ckpt_path} or {dummy_image_path} does not exist.")

    # Load model and class names
    model = load_model(ckpt_path=dummy_ckpt_path, num_classes=num_classes)
    class_names = [f"breed_{i}" for i in range(num_classes)]

    try:
        predicted_class = predict_single_image(model, image_path=dummy_image_path, class_names=class_names)
        assert predicted_class in class_names, "Prediction failed or class not in class names."
    except Exception as e:
        pytest.fail(f"Prediction failed with exception: {e}")


# import pytest
# from pathlib import Path
# import sys
# import lightning as L
# # Add the src directory to sys.path using a relative path
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# # Import infer from the src directory
# from src.infer import load_model, predict_single_image, load_class_names
# from PIL import Image
# from unittest.mock import patch

# @pytest.fixture
# def mock_sys_args():
#     # Backup and mock the sys.argv
#     original_args = sys.argv
#     sys.argv = ["pytest"]
#     yield
#     sys.argv = original_args

# def test_load_model():
#     # Use relative path to checkpoint file
#     dummy_ckpt_path = Path("../checkpoints/mobilenetv2_checkpoint.ckpt")
#     num_classes = 10

#     if not dummy_ckpt_path.exists():
#         pytest.skip(f"Skipping test: {dummy_ckpt_path} does not exist.")

#     try:
#         model = load_model(ckpt_path=dummy_ckpt_path, num_classes=num_classes)
#     except Exception as e:
#         pytest.fail(f"Model loading failed with exception: {e}")

# def test_load_class_names():
#     # Use relative path to dataset directory
#     dummy_data_dir = Path("../data/dataset")

#     if not dummy_data_dir.exists():
#         pytest.skip(f"Skipping test: {dummy_data_dir} does not exist.")

#     try:
#         class_names = load_class_names(dummy_data_dir)
#         assert len(class_names) > 0, "Class names not loaded correctly."
#     except Exception as e:
#         pytest.fail(f"Failed to load class names with exception: {e}")

# def test_predict_single_image():
#     # Use relative paths for checkpoint and image
#     dummy_ckpt_path = Path("../checkpoints/mobilenetv2_checkpoint.ckpt")
#     dummy_image_path = Path("../new_images/image2.jpg")
#     num_classes = 10

#     if not dummy_ckpt_path.exists() or not dummy_image_path.exists():
#         pytest.skip(f"Skipping test: {dummy_ckpt_path} or {dummy_image_path} does not exist.")

#     model = load_model(ckpt_path=dummy_ckpt_path, num_classes=num_classes)
#     class_names = [f"class_{i}" for i in range(num_classes)]

#     try:
#         predicted_class = predict_single_image(model, image_path=dummy_image_path, class_names=class_names)
#         assert predicted_class in class_names, "Prediction failed or class not in class names."
#     except Exception as e:
#         pytest.fail(f"Prediction failed with exception: {e}")
