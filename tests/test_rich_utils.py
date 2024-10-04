import pytest

import sys
sys.path.insert(0, '../src')

from utils.rich_utils import print_config_tree, print_rich_panel, print_rich_progress
from rich.console import Console
from rich.panel import Panel
import rich.syntax
import rich.tree
#from utils.rich_utils import print_rich_panel, print_rich_progress, print_config_tree

def test_print_rich_panel(capsys):
    print_rich_panel("This is a test message", "Test Panel")
    captured = capsys.readouterr()
    assert "Test Panel" in captured.out

def test_print_rich_progress(capsys):
    print_rich_progress("Test Progress")
    captured = capsys.readouterr()
    assert "Test Progress" in captured.out

def test_print_config_tree():
    config = {"data": {"path": "/data"}, "model": {"name": "resnet"}}
    print_config_tree(config)  # Manually validate the console output if needed


@pytest.fixture
def sample_config():
    return {
        "data": {"batch_size": 32, "num_workers": 4},
        "model": {"hidden_units": 128, "dropout": 0.5},
        "trainer": {"max_epochs": 10},
    }

def test_print_config_tree(capsys, sample_config):
    # Capture the output of print_config_tree
    print_config_tree(sample_config)
    captured = capsys.readouterr()
    assert "data" in captured.out, "Output should contain 'data' key from config"
    assert "model" in captured.out, "Output should contain 'model' key from config"
    assert "trainer" in captured.out, "Output should contain 'trainer' key from config"

def test_print_rich_panel(capsys):
    # Test the rich panel print
    print_rich_panel("This is a test panel.", title="Test Panel")
    captured = capsys.readouterr()
    assert "This is a test panel." in captured.out, "Output should contain the text in the panel"
    assert "Test Panel" in captured.out, "Output should contain the title of the panel"

def test_print_rich_progress(capsys):
    # Test the rich progress bar print (note: difficult to capture actual progress bar)
    print_rich_progress("Processing data...")
    captured = capsys.readouterr()
    assert "Processing data..." in captured.out, "Output should contain the description of the task"
