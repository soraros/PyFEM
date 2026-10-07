# SPDX-License-Identifier: MIT
"""Exercise the real solver on a two-DOF problem with an exact tangent."""

import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

solver_module = importlib.import_module('pyfem.solvers.DissipatedEnergySolver')


class _Properties(SimpleNamespace):
  def __iter__(self):
    return iter(vars(self).items())


class _Status:
  def __init__(self):
    self.cycle = 0
    self.iiter = 0

  def increaseStep(self):
    self.cycle += 1
    self.iiter = 0


class _Dofs:
  def __len__(self):
    return 2

  solve = staticmethod(np.linalg.solve)
  norm = staticmethod(np.linalg.norm)


def _assemble_tangent(props, globdat):
  # Gradient/Hessian of x^2 - xy + y^2/2 + y^4/4.
  # The tangent is positive definite for every real y.
  x, y = globdat.state
  tangent = np.array([[2.0, -1.0], [-1.0, 1.0 + 3.0 * y**2]])
  force = np.array([2.0 * x - y, y + y**3 - x])
  return tangent, force


class TestDissipatedEnergySolverLimit(unittest.TestCase):
  def setUp(self):
    self.props = _Properties(
      currentModule='solver',
      solver=_Properties(
        tol=1.0e-10, iterMax=10, maxCycle=10, switchEnergy=float('inf')
      ),
    )
    self.globdat = SimpleNamespace(
      solverStatus=_Status(),
      dofs=_Dofs(),
      elements=SimpleNamespace(commitHistory=Mock()),
      state=np.zeros(2),
      Dstate=np.zeros(2),
      active=True,
    )
    replacements = {
      'assembleExternalForce': lambda props, globdat: np.array([1.0, 0.0]),
      'assembleTangentStiffness': _assemble_tangent,
      'assembleDissipation': lambda props, globdat: (np.zeros(2), 0.0),
    }
    for name, replacement in replacements.items():
      patcher = patch.object(solver_module, name, replacement)
      patcher.start()
      self.addCleanup(patcher.stop)

  def _solver(self, limit):
    self.props.solver.iterMax = limit
    solver = solver_module.DissipatedEnergySolver(self.props, self.globdat)
    self.assertEqual(solver.iterMax, limit)
    for name in (
      'writeHeader', 'writeFooter', 'printHeader', 'printIteration', 'printConverged'
    ):
      setattr(solver, name, Mock())
    return solver

  def _assert_limit(self, limit):
    solver = self._solver(limit)
    with self.assertRaisesRegex(RuntimeError, 'did not converge'):
      solver.run(self.props, self.globdat)
    self.assertEqual(self.globdat.solverStatus.iiter, limit)
    self.globdat.elements.commitHistory.assert_not_called()
    solver.printConverged.assert_not_called()

  def test_stops_after_one_unconverged_iteration(self):
    self._assert_limit(1)

  def test_stops_at_configured_limit_without_committing(self):
    self._assert_limit(2)

  def _assert_nonlinear_solution(self):
    x, y = self.globdat.state
    self.assertAlmostEqual(x + y, 2.0, places=10)
    self.assertAlmostEqual(y**3 + 2.0 * y, 2.0, places=10)
    self.assertAlmostEqual(self.globdat.lam, 2.0 * x - y, places=10)
    self.globdat.elements.commitHistory.assert_called_once_with()

  def test_accepts_convergence_on_last_allowed_iteration(self):
    self._solver(4).run(self.props, self.globdat)
    self.assertEqual(self.globdat.solverStatus.iiter, 4)
    self._assert_nonlinear_solution()

  def test_accepts_convergence_before_limit(self):
    self._solver(10).run(self.props, self.globdat)
    self.assertEqual(self.globdat.solverStatus.iiter, 4)
    self._assert_nonlinear_solution()

  def test_accepts_linear_problem_with_limit_one(self):
    tangent = np.diag([2.0, 3.0])
    with patch.object(
      solver_module,
      'assembleTangentStiffness',
      lambda props, globdat: (tangent, tangent @ globdat.state),
    ):
      self._solver(1).run(self.props, self.globdat)
    self.assertEqual(self.globdat.solverStatus.iiter, 1)
    np.testing.assert_allclose(self.globdat.state, [0.5, 0.0])
    self.globdat.elements.commitHistory.assert_called_once_with()


if __name__ == '__main__':
  unittest.main()
