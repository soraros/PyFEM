# SPDX-License-Identifier: MIT
# Copyright (c) 2011-2026 Joris J.C. Remmers

import unittest

import numpy as np

from pyfem.elements.PhaseField import PhaseField
from pyfem.util.dataStructures import Properties, elementData, solverStatus


class PhaseFieldTestCase(unittest.TestCase):
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

class TestPhaseFieldHistory(PhaseFieldTestCase):
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


class TestPhaseFieldTangent(PhaseFieldTestCase):
    def get_tangent_and_finite_difference(self):
        element = self.make_element()
        tangent_data = self.make_data(1.0)
        element.mat.reset()
        element.getTangentStiffness(tangent_data)

        reference_force = self.evaluate_internal_force(
            self.make_element(), self.make_data(1.0)
        )
        np.testing.assert_array_equal(tangent_data.fint, reference_force)

        finite_difference = np.zeros((12, 12))
        step = 1.0e-7
        for column in range(12):
            plus = self.make_data(1.0)
            minus = self.make_data(1.0)
            plus.state[column] += step
            minus.state[column] -= step
            force_plus = self.evaluate_internal_force(element, plus)
            force_minus = self.evaluate_internal_force(element, minus)
            finite_difference[:, column] = (force_plus - force_minus) / (2.0 * step)

        return element, tangent_data.stiff, finite_difference

    def test_all_rectangular_blocks_match_central_differences(self):
        element, tangent, finite_difference = self.get_tangent_and_finite_difference()
        u_dofs, p_dofs = element.splitDofIDs(4)
        blocks = [
            ("u,u", u_dofs, u_dofs),
            ("u,phase", u_dofs, p_dofs),
            ("phase,u", p_dofs, u_dofs),
            ("phase,phase", p_dofs, p_dofs),
        ]

        for name, rows, columns in blocks:
            with self.subTest(block=name):
                indices = np.ix_(rows, columns)
                np.testing.assert_allclose(
                    tangent[indices],
                    finite_difference[indices],
                    rtol=5.0e-6,
                    atol=5.0e-5,
                )

    def test_legacy_transposed_coupling_is_a_failing_control(self):
        element, tangent, finite_difference = self.get_tangent_and_finite_difference()
        u_dofs, p_dofs = element.splitDofIDs(4)
        phase_u = np.ix_(p_dofs, u_dofs)
        u_phase = np.ix_(u_dofs, p_dofs)
        tangent[phase_u] = tangent[u_phase].transpose()

        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(
                tangent[phase_u],
                finite_difference[phase_u],
                rtol=5.0e-6,
                atol=5.0e-5,
            )


if __name__ == "__main__":
    unittest.main()
