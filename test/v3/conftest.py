# SPDX-License-Identifier: MIT

"""Shared fixtures for v3 tests."""

from __future__ import annotations

import sys

import pytest

pytestmark = pytest.mark.skipif(
  sys.version_info < (3, 13),
  reason="pyfem.v3 requires Python 3.13+",
)
