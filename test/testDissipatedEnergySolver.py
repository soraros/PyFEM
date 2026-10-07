# SPDX-License-Identifier: MIT

"""Iteration-limit regressions with a two-DOF, consistent-tangent fixture.

Only assembly is replaced; the actual solver constructor, Newton iteration,
convergence test and history-commit decision run unchanged.
"""

from importlib import import_module
from time import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from pyfem.util.dataStructures import Properties, solverStatus

solver_module = import_module('pyfem.solvers.DissipatedEnergySolver')


class TwoDofs:
  def __len__(self):
    return 2

  def solve(self, matrix, rhs):
    return np.linalg.solve(matrix, rhs)

  def norm(self, vector):
    return float(np.linalg.norm(vector))


def tangent(props, globdat):
  """f_int = (x-y, 2y+y^2-x), with its exact Jacobian."""
  x, y = globdat.state
  return (
    np.array([[1.0, -1.0], [-1.0, 2.0 + 2.0 * y]]),
    np.array([x - y, 2.0 * y + y * y - x]),
  )


def control(props, globdat):
  """A linear continuation functional, not a constitutive dissipation model."""
  return np.array([1.0, 0.0]), float(globdat.state[0])


class TestDissipatedEnergySolverIterations(unittest.TestCase):
  def setUp(self):
    self.patch_assembly('assembleExternalForce', return_value=np.array([1.0, 0.0]))
    self.assembly = self.patch_assembly(
      'assembleTangentStiffness', side_effect=tangent
    )
    self.patch_assembly('assembleDissipation', side_effect=control)

  def patch_assembly(self, name, **kwargs):
    patcher = patch.object(solver_module, name, **kwargs)
    self.addCleanup(patcher.stop)
    return patcher.start()

  def make_solver(self, limit, method):
    props = Properties({
      'currentModule': 'solver',
      'solver': Properties({
        'iterMax': limit,
        'tol': 1.0e-7,
        'switchEnergy': float('inf'),
        'maxCycle': 2,
      }),
    })
    globdat = SimpleNamespace(
      state=np.zeros(2),
      Dstate=np.zeros(2),
      dofs=TwoDofs(),
      solverStatus=solverStatus(),
      elements=SimpleNamespace(commitHistory=Mock()),
      startTime=time(),
      active=True,
    )
    solver = solver_module.DissipatedEnergySolver(props, globdat)
    self.assertEqual(solver.iterMax, limit)
    solver.method = method
    globdat.dtau = 2.0
    return props, globdat, solver

  def test_iteration_limit_is_enforced_before_history_commit(self):
    for method in ('arclength-controlled', 'nrg-controlled'):
      with self.subTest(method=method):
        props, globdat, solver = self.make_solver(2, method)
        with self.assertRaisesRegex(RuntimeError, 'iterations did not converge'):
          solver.run(props, globdat)
        self.assertEqual(globdat.solverStatus.iiter, 2)
        globdat.elements.commitHistory.assert_not_called()

  def test_convergence_on_last_allowed_iteration_is_accepted(self):
    for method in ('arclength-controlled', 'nrg-controlled'):
      with self.subTest(method=method):
        props, globdat, solver = self.make_solver(3, method)
        solver.run(props, globdat)
        self.assertEqual(globdat.solverStatus.iiter, 3)
        globdat.elements.commitHistory.assert_called_once_with()
        _, fint = tangent(props, globdat)
        residual = globdat.lam * np.array([1.0, 0.0]) - fint
        self.assertLessEqual(
          globdat.dofs.norm(residual) / abs(globdat.lam), solver.tol
        )

  def test_convergence_before_limit_is_accepted(self):
    for method in ('arclength-controlled', 'nrg-controlled'):
      with self.subTest(method=method):
        props, globdat, solver = self.make_solver(4, method)
        solver.run(props, globdat)
        self.assertEqual(globdat.solverStatus.iiter, 3)
        globdat.elements.commitHistory.assert_called_once_with()

  def test_convergence_with_single_allowed_iteration_is_accepted(self):
    matrix = np.array([[1.0, -1.0], [-1.0, 2.0]])
    self.assembly.side_effect = lambda props, globdat: (
      matrix, matrix @ globdat.state
    )
    for method in ('arclength-controlled', 'nrg-controlled'):
      with self.subTest(method=method):
        props, globdat, solver = self.make_solver(1, method)
        solver.run(props, globdat)
        self.assertEqual(globdat.solverStatus.iiter, 1)
        globdat.elements.commitHistory.assert_called_once_with()


if __name__ == '__main__':
  unittest.main()
