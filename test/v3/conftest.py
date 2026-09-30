# SPDX-License-Identifier: MIT

"""Shared fixtures for v3 tests."""

from __future__ import annotations

import ast
import io
import sys
import tokenize
from collections.abc import Callable
from pathlib import Path

import pytest

_V3_TEST = Path(__file__).resolve().parent
if str(_V3_TEST) not in sys.path:
  sys.path.insert(0, str(_V3_TEST))

pytestmark = pytest.mark.skipif(
  sys.version_info < (3, 13),
  reason="pyfem.v3 requires Python 3.13+",
)

_REPO_ROOT = _V3_TEST.parents[1]
_V3_FEM_DIR = _REPO_ROOT / "pyfem" / "v3" / "fem"

# Cache-policy canary (NUMBA_CACHING.md §5): an @njit(cache=True) kernel whose
# body references names imported from other pyfem.v3 modules is a bug — numba's
# on-disk cache cannot invalidate cross-module callees or frozen globals. Such
# sites must instead carry an explicit cache=False with a comment citing
# NUMBA_CACHING.md. The scan is AST/tokenize-only and never touches cache state.

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


@pytest.fixture(scope="session", autouse=True)
def _numba_cache_policy_canary(
  numba_cache_policy_scanner: Callable[[Path], list[str]],
) -> None:
  """Fail the test session on cache-unfriendly kernels in pyfem/v3/fem."""
  violations = numba_cache_policy_scanner(_V3_FEM_DIR)
  if violations:
    pytest.fail(
      "numba cache policy violations in pyfem/v3/fem (NUMBA_CACHING.md §5):\n"
      + "\n".join(f"  - {violation}" for violation in violations),
      pytrace=False,
    )
