# SPDX-License-Identifier: MIT
"""Regression tests for trial/committed dissipation in XuNeedleman."""

import math
import unittest
from types import SimpleNamespace

import numpy as np

from pyfem.materials.XuNeedleman import XuNeedleman


class _Properties(SimpleNamespace):
  def __iter__(self):
    return iter(vars(self).items())


def _material():
  return XuNeedleman(_Properties(rank=2, Gc=0.2, Tult=40.0, solverStat=None))


def _evaluate(material, opening):
  """Opening is dimensionless: normal separation divided by vnmax."""
  kin = SimpleNamespace(strain=np.array([opening * material.vnmax, 0.0]))
  stress, tangent = material.getStress(kin)
  return kin, stress, tangent


def _dissipation(material, opening):
  """Independent pure-mode-I expression for q=1, r=0."""
  return material.Gc * (
    1.0 - (1.0 + opening + 0.5 * opening**2) * math.exp(-opening)
  )


class TestXuNeedlemanHistory(unittest.TestCase):
  def test_rejected_loading_trial_does_not_survive_unloading(self):
    material = _material()
    _evaluate(material, 1.0)
    material.commitHistory()
    committed = material.getHistoryParameter('dissipation')

    _evaluate(material, 4.0)  # Unaccepted Newton trial; do not commit.
    kin, _, _ = _evaluate(material, 0.5)

    self.assertEqual(kin.g, 0.0)
    np.testing.assert_array_equal(kin.dgdstrain, np.zeros(2))
    material.commitHistory()
    self.assertEqual(material.getHistoryParameter('dissipation'), committed)

  def test_trial_evaluations_leave_committed_history_unchanged(self):
    material = _material()
    _evaluate(material, 1.0)
    material.commitHistory()
    committed = material.getHistoryParameter('dissipation')
    for opening in (4.0, 0.5, 2.0, 0.0):
      _evaluate(material, opening)
      self.assertEqual(material.getHistoryParameter('dissipation'), committed)

  def test_monotone_loading_keeps_existing_dissipation_law(self):
    material = _material()
    previous = 0.0
    for opening in (0.5, 1.0, 2.0, 4.0):
      kin, _, _ = _evaluate(material, opening)
      expected = _dissipation(material, opening)
      self.assertAlmostEqual(kin.g, expected - previous, places=14)
      material.commitHistory()
      self.assertAlmostEqual(
        material.getHistoryParameter('dissipation'), expected, places=14
      )
      previous = expected

  def test_return_to_committed_opening_discards_overshoot(self):
    material = _material()
    _evaluate(material, 1.0)
    material.commitHistory()
    committed = material.getHistoryParameter('dissipation')
    _evaluate(material, 4.0)
    _evaluate(material, 1.0)
    material.commitHistory()
    self.assertEqual(material.getHistoryParameter('dissipation'), committed)

  def test_smaller_accepted_loading_trial_replaces_overshoot(self):
    material = _material()
    _evaluate(material, 1.0)
    material.commitHistory()
    _evaluate(material, 4.0)
    _evaluate(material, 2.0)
    material.commitHistory()
    self.assertAlmostEqual(
      material.getHistoryParameter('dissipation'),
      _dissipation(material, 2.0),
      places=14,
    )


if __name__ == '__main__':
  unittest.main()
