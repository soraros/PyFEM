"""Prescribed-displacement constraints."""

from __future__ import annotations

from pyfem.fem.Constrainer import Constrainer
from pyfem.v3.types import ProblemDefinition


def build_constrainer(problem: ProblemDefinition) -> Constrainer:
  """Build legacy ``Constrainer`` from packed constraint arrays."""
  cons = Constrainer(problem.n_dofs)
  label = "main"
  cons.constrainedDofs[label] = []
  cons.constrainedVals[label] = []
  cons.constrainedFac[label] = 1.0

  for dof_id, value in zip(problem.constraint_dof, problem.constraint_val):
    cons.addConstraint(int(dof_id), float(value), label)

  cons.flush()
  return cons
