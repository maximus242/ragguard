"""
Pytest configuration for RAGGuard tests.
"""

import pytest


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--stability-full",
        action="store_true",
        default=False,
        help="Run extended stability tests (takes ~1 minute)"
    )
