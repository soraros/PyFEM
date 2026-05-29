# SPDX-License-Identifier: MIT

"""Shared fixtures for v3 tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_V3_TEST = Path(__file__).resolve().parent
if str(_V3_TEST) not in sys.path:
  sys.path.insert(0, str(_V3_TEST))

pytestmark = pytest.mark.skipif(
  sys.version_info < (3, 13),
  reason="pyfem.v3 requires Python 3.13+",
)
