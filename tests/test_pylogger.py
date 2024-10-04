import pytest
import logging
from src.utils.pylogger import get_pylogger

def test_get_pylogger():
    # Test logger initialization with default name
    logger = get_pylogger()
    assert isinstance(logger, logging.Logger), "Returned object should be a Logger instance."

def test_logger_with_custom_name():
    # Test logger initialization with a custom name
    logger = get_pylogger("custom_logger")
    assert logger.name == "custom_logger", "Logger name should match the custom name provided."

def test_logger_rank_zero_only():
    # Test if logger methods are wrapped with rank_zero_only decorator
    logger = get_pylogger()
    assert hasattr(logger, 'info'), "Logger should have 'info' attribute."
    assert hasattr(logger.info, "__wrapped__"), "Logger's info method should be wrapped by rank_zero_only."
