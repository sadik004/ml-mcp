import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.smoke_test import run_smoke_test


def test_smoke_test_script_execution():
    """Verify that smoke_test.py executes cleanly without throwing any exceptions."""
    run_smoke_test()
