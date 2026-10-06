# SPDX-License-Identifier: MIT

"""Shared fixtures for v3 tests."""

from __future__ import annotations

import ast
import io
import platform
import sys
import tokenize
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

_V3_TEST = Path(__file__).resolve().parent
if str(_V3_TEST) not in sys.path:
  sys.path.insert(0, str(_V3_TEST))

pytestmark = pytest.mark.skipif(
  sys.version_info < (3, 13),
  reason="pyfem.v3 requires Python 3.13+",
)

_REPO_ROOT = _V3_TEST.parents[1]
_V3_DIR = _REPO_ROOT / "pyfem" / "v3"

# Cache-policy canary (NUMBA_CACHING.md §5): an @njit(cache=True) kernel whose
# body references names imported from other pyfem.v3 modules is a bug — numba's
# on-disk cache cannot invalidate cross-module callees or frozen globals. Such
# sites must instead carry an explicit cache=False with a comment citing
# NUMBA_CACHING.md. The scan is AST/tokenize-only and never touches cache state.
#
# Scan roots: the pyfem.v3 packages that host @njit kernels — currently fem/
# and materials/ (M30's J2 return-map kernels). When another pyfem.v3 package
# grows @njit kernels, add its directory name to _CACHE_POLICY_SCAN_PACKAGES
# and to _SELF_TEST_EXPECTED_ROOTS below, so the policy covers it from the
# day the kernels land.

_CACHE_POLICY_SCAN_PACKAGES: tuple[str, ...] = ("fem", "materials")
_CACHE_POLICY_SCAN_ROOTS: tuple[Path, ...] = tuple(
  _V3_DIR / package for package in _CACHE_POLICY_SCAN_PACKAGES
)

_DOC_MARKER = "NUMBA_CACHING"
_DOC_WINDOW_LINES = 8


def _package_dotted(path: Path) -> str:
  """Dotted package of ``path`` per ``__init__.py`` parents ("" if unpackaged)."""
  parts: list[str] = []
  parent = path.parent
  while (parent / "__init__.py").is_file():
    parts.append(parent.name)
    parent = parent.parent
  return ".".join(reversed(parts))


def _resolve_import_from(module: str | None, level: int, package: str) -> str:
  if not level:
    return module or ""
  base = package.split(".") if package else []
  if level > 1:
    base = base[: 1 - level]
  return ".".join([*base, module] if module else base)


def _v3_imported_names(tree: ast.Module, package: str) -> dict[str, str]:
  """Map module-level names to the pyfem.v3 module they were imported from."""
  imported: dict[str, str] = {}
  for node in tree.body:
    if isinstance(node, ast.Import):
      for alias in node.names:
        if alias.name.startswith("pyfem.v3"):
          imported[alias.asname or alias.name.partition(".")[0]] = alias.name
    elif isinstance(node, ast.ImportFrom):
      source = _resolve_import_from(node.module, node.level, package)
      if source.startswith("pyfem.v3"):
        for alias in node.names:
          if alias.name != "*":
            imported[alias.asname or alias.name] = source
  return imported


def _njit_cache_flag(deco: ast.expr) -> bool | str | None:
  """cache kwarg of an ``njit`` decorator: literal bool, "default", or None."""
  if not isinstance(deco, ast.Call):
    return None
  func = deco.func
  if isinstance(func, ast.Name):
    is_njit = func.id == "njit"
  elif isinstance(func, ast.Attribute):
    is_njit = func.attr == "njit"
  else:
    is_njit = False
  if not is_njit:
    return None
  for keyword in deco.keywords:
    if keyword.arg == "cache":
      value = keyword.value
      if isinstance(value, ast.Constant) and isinstance(value.value, bool):
        return value.value
      return "default"
  return "default"


class _BodyReferences(ast.NodeVisitor):
  """Collect (name, lineno) loads in a kernel body, skipping type annotations."""

  def __init__(self) -> None:
    self.names: set[tuple[str, int]] = set()

  def visit_Name(self, node: ast.Name) -> None:
    if isinstance(node.ctx, ast.Load):
      self.names.add((node.id, node.lineno))

  def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
    self.visit(node.target)
    if node.value is not None:
      self.visit(node.value)

  def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
    for stmt in node.body:
      self.visit(stmt)

  def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
    for stmt in node.body:
      self.visit(stmt)

  def visit_Lambda(self, node: ast.Lambda) -> None:
    self.visit(node.body)


def _kernel_references(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
  visitor = _BodyReferences()
  for stmt in func.body:
    visitor.visit(stmt)
  return {name for name, _ in visitor.names}


def _comment_lines(source: str) -> dict[int, str]:
  comments: dict[int, str] = {}
  for token in tokenize.generate_tokens(io.StringIO(source).readline):
    if token.type == tokenize.COMMENT:
      comments[token.start[0]] = token.string
  return comments


def _is_documented(comments: dict[int, str], deco_lineno: int) -> bool:
  start = max(1, deco_lineno - _DOC_WINDOW_LINES)
  return any(
    _DOC_MARKER in comments.get(line, "") for line in range(start, deco_lineno)
  )


def scan_numba_cache_policy(root: Path) -> list[str]:
  """Return NUMBA_CACHING.md §5 violations under ``root`` (empty list = clean).

  A violation is an ``njit`` kernel whose body references a name imported from
  another pyfem.v3 module, unless the kernel carries an explicit ``cache=False``
  with a comment citing NUMBA_CACHING.md above the decorator.
  """
  violations: list[str] = []
  for path in sorted(root.rglob("*.py")):
    if "__pycache__" in path.parts:
      continue
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imported = _v3_imported_names(tree, _package_dotted(path))
    if not imported:
      continue
    comments = _comment_lines(source)
    for node in tree.body:
      if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        continue
      flags = [
        (flag, deco.lineno)
        for deco in node.decorator_list
        if (flag := _njit_cache_flag(deco)) is not None
      ]
      if not flags:
        continue
      refs = {
        name: imported[name] for name in _kernel_references(node) if name in imported
      }
      if not refs:
        continue
      cache_flag, deco_lineno = flags[0]
      rel = path.relative_to(root)
      details = ", ".join(f"{name} (from {refs[name]})" for name in sorted(refs))
      if cache_flag is True:
        violations.append(
          f"{rel}:{node.lineno}: cache=True kernel '{node.name}' references "
          f"cross-module pyfem.v3 names: {details} — numba's on-disk cache "
          f"cannot invalidate these (NUMBA_CACHING.md §5)"
        )
      elif cache_flag == "default":
        violations.append(
          f"{rel}:{node.lineno}: kernel '{node.name}' references cross-module "
          f"pyfem.v3 names: {details} without an explicit documented cache=False "
          f"(NUMBA_CACHING.md §5)"
        )
      elif not _is_documented(comments, deco_lineno):
        violations.append(
          f"{rel}:{node.lineno}: cache=False kernel '{node.name}' references "
          f"cross-module pyfem.v3 names: {details} but is not a documented "
          f"cache=False site (comment citing NUMBA_CACHING.md required)"
        )
  return violations


@pytest.fixture(scope="session")
def numba_cache_policy_scanner() -> Callable[[Path], list[str]]:
  """The cache-policy AST scanner (NUMBA_CACHING.md §5); reads source only."""
  return scan_numba_cache_policy


def _scan_cache_policy_roots(
  scanner: Callable[[Path], list[str]],
  roots: tuple[Path, ...],
) -> list[str]:
  """Violations across all scan ``roots``, each prefixed with its repo path."""
  violations: list[str] = []
  for root in roots:
    prefix = root.relative_to(_REPO_ROOT)
    violations.extend(f"{prefix}/{violation}" for violation in scanner(root))
  return violations


_PLANTED_VIOLATION_NAME = "_canary_planted_violation.py"
_PLANTED_VIOLATION_SOURCE = """\
from numba import njit

from pyfem.v3.fem.quadrature import gauss_tria3


@njit(cache=True)
def planted_cached() -> tuple:
  return gauss_tria3(1)
"""


# Roots the self-test plants under, hardcoded apart from
# _CACHE_POLICY_SCAN_PACKAGES on purpose: deriving them from the constant the
# canary reads would let a dropped package pass silently. Keep both lists in
# sync when a kernel-hosting package joins or leaves.
_SELF_TEST_EXPECTED_ROOTS: tuple[Path, ...] = (
  _V3_DIR / "fem",
  _V3_DIR / "materials",
)


@pytest.fixture(scope="session", autouse=True)
def _numba_cache_policy_canary_self_test(
  numba_cache_policy_scanner: Callable[[Path], list[str]],
) -> None:
  """Prove the canary fires on a planted violation under each expected root.

  Plants a NUMBA_CACHING.md §5 violation in every kernel-hosting package in
  _SELF_TEST_EXPECTED_ROOTS (currently fem/ and materials/) and requires
  the canary's scan to flag each one: a package dropped from
  _CACHE_POLICY_SCAN_PACKAGES leaves its planted violation undetected.
  """
  planted = [root / _PLANTED_VIOLATION_NAME for root in _SELF_TEST_EXPECTED_ROOTS]
  try:
    for path in planted:
      path.write_text(_PLANTED_VIOLATION_SOURCE, encoding="utf-8")
    violations = _scan_cache_policy_roots(
      numba_cache_policy_scanner, _CACHE_POLICY_SCAN_ROOTS
    )
  finally:
    for path in planted:
      path.unlink(missing_ok=True)
  missed = [
    str(path.relative_to(_REPO_ROOT))
    for path in planted
    if not any(str(path.relative_to(_REPO_ROOT)) in hit for hit in violations)
  ]
  if missed:
    pytest.fail(
      "numba cache policy canary missed planted violations (NUMBA_CACHING.md "
      "§5 self-test — is every kernel-hosting package still scanned?):\n"
      + "\n".join(f"  - {path}" for path in missed),
      pytrace=False,
    )


@pytest.fixture(scope="session", autouse=True)
def _numba_cache_policy_canary(
  numba_cache_policy_scanner: Callable[[Path], list[str]],
) -> None:
  """Fail the test session on cache-unfriendly kernels under the scan roots.

  Scan roots: pyfem/v3/fem and pyfem/v3/materials — see
  _CACHE_POLICY_SCAN_PACKAGES for how future kernel-hosting packages join.
  """
  violations = _scan_cache_policy_roots(
    numba_cache_policy_scanner, _CACHE_POLICY_SCAN_ROOTS
  )
  if violations:
    pytest.fail(
      "numba cache policy violations (NUMBA_CACHING.md §5):\n"
      + "\n".join(f"  - {violation}" for violation in violations),
      pytrace=False,
    )


# ---------------------------------------------------------------------------
# Reference-platform contract for bitwise numerical pins.
#
# Byte-identity pins (raw-uint64 equality, signed-zero distinctions included)
# are defined on ONE reference platform: the machine the bench manifests
# record (bench/manifest.py collect_manifest, bench/results/*.json) — macOS
# on x86_64 (Intel i9-9980HK), Python 3.13, the pyproject-pinned numba
# (>=0.62.1,<0.63) and numpy, Accelerate BLAS. Off that platform the same
# physics holds but the bits need not: numpy's non-macOS wheels bundle
# OpenBLAS instead of Accelerate, and LLVM lowers numba kernels per target,
# so BLAS path selection and codegen round differently at ulp level.
# Observed on ubuntu-latest CI (Python 3.13, numba 0.62.1, numpy 2.3.5; runs
# 37440534665 and 37441926273): the generic-core spring residual term landed
# 4 ulps off ±1.0 (8.9e-16), and the M30 plasticity kernel's stresses drifted
# up to 32 ulps (~7.1e-15 relative) from the M25 reference.
#
# The contract: a bitwise pin asserts raw-uint64 identity on the reference
# platform and a documented tight tolerance everywhere else. BOTH branches
# always assert — the platform selects the comparison, it never skips it.
# Each call site's tolerance must exceed the observed cross-platform
# deviation, with that deviation cited in a comment at the call site.

REFERENCE_PLATFORM = (
  "macOS x86_64 (bench-manifest reference: Intel i9-9980HK, numba 0.62.1, "
  "pinned numpy, Accelerate BLAS)"
)
"""Human-readable description of the bitwise-reference platform."""


def on_reference_platform() -> bool:
  """True on the bitwise-reference platform (see the contract above).

  Reads ``sys.platform``/``platform.machine()`` at call time so tests can
  force either branch with ``monkeypatch``.
  """
  return sys.platform == "darwin" and platform.machine() == "x86_64"


def assert_bitwise_pin(
  actual: np.ndarray,
  expected: np.ndarray,
  *,
  rtol: float,
  atol: float,
) -> None:
  """Assert the bitwise-pin contract for float64 arrays.

  On the reference platform: raw-uint64 identity (signed-zero distinctions
  included). Anywhere else: ``assert_allclose`` with the call site's
  documented tolerance. Both branches always assert.
  """
  if on_reference_platform():
    np.testing.assert_array_equal(
      np.asarray(actual, dtype=np.float64).view(np.uint64),
      np.asarray(expected, dtype=np.float64).view(np.uint64),
      err_msg=(
        f"byte-identity pin failed on the reference platform ({REFERENCE_PLATFORM})"
      ),
    )
  else:
    np.testing.assert_allclose(
      actual,
      expected,
      rtol=rtol,
      atol=atol,
      err_msg=(
        "bitwise pin exceeded its documented cross-platform tolerance "
        f"({sys.platform}/{platform.machine()}, rtol={rtol}, atol={atol}; "
        f"reference: {REFERENCE_PLATFORM})"
      ),
    )


@pytest.fixture(scope="session")
def bitwise_pin() -> Callable[..., None]:
  """The ``assert_bitwise_pin`` contract assert (see its docstring)."""
  return assert_bitwise_pin


@pytest.fixture(scope="session")
def reference_platform_probe() -> Callable[[], bool]:
  """The ``on_reference_platform`` probe as a function, not a bool.

  Tests monkeypatch ``sys.platform``/``platform.machine()`` and then call the
  probe to prove the gate follows — a resolved bool could not.
  """
  return on_reference_platform
