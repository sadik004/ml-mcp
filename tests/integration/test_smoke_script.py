"""Integration test executing scripts/smoke_test.py via pytest."""
import pytest
from scripts.smoke_test import run_smoke_test


def test_smoke_test_script_execution():
    """Verify that smoke_test.py executes cleanly without throwing any exceptions."""
    run_smoke_test()
