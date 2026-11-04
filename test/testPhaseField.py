# SPDX-License-Identifier: MIT
# Copyright (c) 2011-2026 Joris J.C. Remmers

import unittest

import numpy as np

from pyfem.elements.PhaseField import PhaseField
from pyfem.util.dataStructures import Properties, elementData, solverStatus


class TestPhaseFieldHistory(unittest.TestCase):
    def setUp(self):
        self.coords = np.array(
            [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]
        )
        self.phase = np.array([0.15, 0.25, 0.35, 0.2])

    def make_element(self):
        props = Properties(
            {
                "rank": 2,
                "solverStat": solverStatus(),
                "material": Properties(
                    {"type": "PlaneStress", "E": 2000.0, "nu": 0.25}
                ),
                "Gc": 1.5,
                "l0": 0.2,
            }
        )
        element = PhaseField([0, 1, 2, 3], props)
        element.appendNodalOutput = lambda *args: None
        return element

    def make_data(self, strain_scale):
        state = np.zeros(12)
        p_dofs = [2, 5, 8, 11]
        exx = 0.004 * strain_scale
        eyy = -0.001 * strain_scale
        gamma = 0.0024 * strain_scale
        for i, (x, y) in enumerate(self.coords):
            state[3 * i] = exx * x + 0.5 * gamma * y
            state[3 * i + 1] = 0.5 * gamma * x + eyy * y
        state[p_dofs] = self.phase
        data = elementData(state, np.zeros(12))
        data.coords = self.coords
        return data

    def evaluate_internal_force(self, element, data):
        element.mat.reset()
        data.fint.fill(0.0)
        element.getInternalForce(data)
        return data.fint.copy()

    def test_rejected_trial_does_not_change_accepted_restart(self):
        element = self.make_element()
        accepted_data = self.make_data(0.25)
        rejected_data = self.make_data(2.0)

        self.evaluate_internal_force(element, accepted_data)
        element.commit(accepted_data)
        accepted_history = element.hisOld.copy()
        self.assertFalse(np.shares_memory(element.hisOld, element.hisNew))

        accepted_force = self.evaluate_internal_force(element, accepted_data)
        self.evaluate_internal_force(element, rejected_data)

        np.testing.assert_array_equal(element.hisOld, accepted_history)

        restarted_force = self.evaluate_internal_force(element, accepted_data)
        np.testing.assert_array_equal(restarted_force, accepted_force)
        np.testing.assert_array_equal(element.hisNew, accepted_history)

        self.evaluate_internal_force(element, rejected_data)
        element.commit(rejected_data)
        accepted_history = element.hisOld.copy()
        accepted_force = self.evaluate_internal_force(element, rejected_data)
        self.evaluate_internal_force(element, self.make_data(4.0))

        np.testing.assert_array_equal(element.hisOld, accepted_history)
        restarted_force = self.evaluate_internal_force(element, rejected_data)
        np.testing.assert_array_equal(restarted_force, accepted_force)
        np.testing.assert_array_equal(element.hisNew, accepted_history)


if __name__ == "__main__":
    unittest.main()
