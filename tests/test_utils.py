import pytest

import sys
sys.path.insert(0, '../src')

from utils.utils import task_wrapper


def test_task_wrapper_success():
    @task_wrapper
    def sample_task(x):
        return x ** 2

    assert sample_task(3) == 9

def test_task_wrapper_exception():
    @task_wrapper
    def faulty_task(x):
        raise ValueError("This is an intentional error.")

    try:
        faulty_task(3)
    except ValueError as e:
        assert str(e) == "This is an intentional error."

def sample_task_success():
    """A sample task function that returns a success message."""
    return "Task completed successfully."

def sample_task_failure():
    """A sample task function that raises an exception."""
    raise ValueError("This is a sample error.")

def test_task_wrapper_success():
    # Test task wrapper with a successful task
    wrapped_task = task_wrapper(sample_task_success)
    result = wrapped_task()
    assert result == "Task completed successfully.", "Wrapped task should return success message."

def test_task_wrapper_exception():
    # Test task wrapper with a failing task
    wrapped_task = task_wrapper(sample_task_failure)
    with pytest.raises(ValueError, match="This is a sample error."):
        wrapped_task()
