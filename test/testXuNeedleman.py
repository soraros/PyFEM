# SPDX-License-Identifier: MIT

"""Trial-history regressions using the actual XuNeedleman material."""

from copy import deepcopy
from math import exp
import unittest

import numpy as np

from pyfem.materials.XuNeedleman import XuNeedleman
from pyfem.util.dataStructures import Properties, solverStatus
from pyfem.util.kinematics import Kinematics


def evaluate(material, normal, shear=0.0):
  """Evaluate a trial jump in units of vnmax and vtmax, respectively."""
  kin = Kinematics(2, 2)
  kin.strain[:] = [normal * material.vnmax, shear * material.vtmax]
  stress, tangent = material.getStress(kin)
  return kin, stress, tangent


class TestXuNeedlemanHistory(unittest.TestCase):
  def setUp(self):
    self.material = XuNeedleman(Properties({
      'rank': 2,
      'Gc': 0.2,  # N/mm
      'Tult': 40.0,  # N/mm^2
      'solverStat': solverStatus(),
    }))

  def test_rejected_trial_is_not_committed_on_unloading(self):
    material = self.material
    evaluate(material, 1.0)
    material.commitHistory()
    accepted = material.getHistoryParameter('dissipation')

    evaluate(material, 4.0)  # Trial only: no commit.
    self.assertGreater(material.newHistory['dissipation'], accepted)
    self.assertEqual(material.getHistoryParameter('dissipation'), accepted)

    kin, _, _ = evaluate(material, 0.5)
    self.assertEqual(kin.g, 0.0)
    np.testing.assert_array_equal(kin.dgdstrain, np.zeros(2))
    material.commitHistory()
    self.assertEqual(material.getHistoryParameter('dissipation'), accepted)

  def test_unloading_is_independent_of_intermediate_trials(self):
    for normal, shear in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
      with self.subTest(normal=normal, shear=shear):
        direct = deepcopy(self.material)
        evaluate(direct, normal, shear)
        direct.commitHistory()
        with_trial = deepcopy(direct)
        evaluate(with_trial, 4.0 * normal, 4.0 * shear)

        expected, stress, tangent = evaluate(direct, 0.5 * normal, 0.5 * shear)
        actual, trial_stress, trial_tangent = evaluate(
          with_trial, 0.5 * normal, 0.5 * shear
        )
        np.testing.assert_array_equal(trial_stress, stress)
        np.testing.assert_array_equal(trial_tangent, tangent)
        np.testing.assert_array_equal(actual.dgdstrain, expected.dgdstrain)
        self.assertEqual(actual.g, expected.g)
        direct.commitHistory()
        with_trial.commitHistory()
        self.assertEqual(with_trial.oldHistory, direct.oldHistory)

  def test_latest_loading_trial_replaces_rejected_trial(self):
    material = self.material
    evaluate(material, 1.0)
    material.commitHistory()
    reference = deepcopy(material)

    evaluate(material, 4.0)  # Overshoot, then accept a smaller loading trial.
    expected, _, _ = evaluate(reference, 2.0)
    actual, _, _ = evaluate(material, 2.0)
    self.assertGreater(actual.g, 0.0)
    self.assertEqual(actual.g, expected.g)
    material.commitHistory()
    reference.commitHistory()
    self.assertEqual(material.oldHistory, reference.oldHistory)

  def test_monotonic_opening_preserves_analytical_dissipation(self):
    material = self.material
    for opening in (0.5, 1.0, 2.0, 4.0):
      with self.subTest(opening=opening):
        old = material.getHistoryParameter('dissipation')
        kin, _, _ = evaluate(material, opening)
        expected = material.Gc * (
          1.0 - exp(-opening) * (1.0 + opening + 0.5 * opening**2)
        )
        self.assertAlmostEqual(kin.g, expected - old, places=14)
        material.commitHistory()
        self.assertAlmostEqual(
          material.getHistoryParameter('dissipation'), expected, places=14
        )


if __name__ == '__main__':
  unittest.main()
