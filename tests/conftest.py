"""Pytest configuration — marks sys so CLI knows it's in test mode."""

import sys

# Signal to the CLI that we're running under pytest
sys._called_from_test = True
