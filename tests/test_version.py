"""Unit tests for __version__.py."""

import ufpdf_xfel_scripts  # noqa


def test_package_version():
    """Ensure the package version is defined and not set to the initial
    placeholder."""
    assert hasattr(ufpdf_xfel_scripts, "__version__")
    assert ufpdf_xfel_scripts.__version__ != "0.0.0"
