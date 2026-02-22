"""Unit tests for __version__.py."""

import bglk_euxfel  # noqa


def test_package_version():
    """Ensure the package version is defined and not set to the initial
    placeholder."""
    assert hasattr(bglk_euxfel, "__version__")
    assert bglk_euxfel.__version__ != "0.0.0"
