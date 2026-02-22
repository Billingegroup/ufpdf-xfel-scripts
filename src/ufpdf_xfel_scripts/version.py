#!/usr/bin/env python
##############################################################################
#
# Copyright (c) 2024-2025, The Trustees of Columbia University in the City of
# New York. All rights reserved.
#
# Copyright (c) 2026-present, The scikit-package developers. All rights
# reserved.
#
# File coded by: Billinge Group members and community contributors.
#
# See GitHub contributions for a more detailed list of contributors.
# https://github.com/scikit-package/scikit-package/graphs/contributors
#
# See LICENSE.rst for license information.
#
##############################################################################
"""Definition of __version__."""

#  We do not use the other three variables, but can be added back if needed.
#  __all__ = ["__date__", "__git_commit__", "__timestamp__", "__version__"]

# obtain version information
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ufpdf-xfel-scripts")
except PackageNotFoundError:
    __version__ = "unknown"
