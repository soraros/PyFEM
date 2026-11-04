# SPDX-License-Identifier: MIT
# Copyright (c) 2011-2026 Joris J.C. Remmers

import unittest

import numpy as np

from pyfem.elements.ThermoSmallStrainContinuum import ThermoSmallStrainContinuum
from pyfem.util.dataStructures import Properties, elementData, solverStatus
from pyfem.util.shapeFunctions import getElemShapeData


class TestThermoSmallStrainContinuum(unittest.TestCase):
    def setUp(self):
        self.coords = np.array(
            [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]
        )
        self.dtime = 0.25

    def make_element(self):
        status = solverStatus()
        status.dtime = self.dtime
        props = Properties(
            {
                "rank": 2,
                "solverStat": status,
                "material": Properties(
                    {
                        "type": "PlaneStress",
                        "E": 2000.0,
                        "nu": 0.25,
                        "heatConductivity": 3.5,
                        "heatCapacity": 7.0,
                        "alpha": 1.2e-4,
                    }
                ),
            }
        )
        element = ThermoSmallStrainContinuum([0, 1, 2, 3], props)
        element.appendNodalOutput = lambda *args: None
        return element

    def make_data(self):
        state = np.zeros(12)
        increment = np.zeros(12)
        temperatures = [10.0, 14.0, 17.0, 12.0]
        temperature_increments = [1.0, -2.0, 0.5, 1.5]
        for i, (x, y) in enumerate(self.coords):
            state[3 * i] = 0.002 * x + 0.0004 * y
            state[3 * i + 1] = 0.0004 * x - 0.0007 * y
            state[3 * i + 2] = temperatures[i]
            increment[3 * i] = 1.0e-4 * (i + 1)
            increment[3 * i + 1] = -5.0e-5 * (i + 1)
            increment[3 * i + 2] = temperature_increments[i]
        data = elementData(state, increment)
        data.coords = self.coords
        return data

    def evaluate_internal_force(self, element, data):
        element.mat.reset()
        data.fint.fill(0.0)
        element.getInternalForce(data)
        return data.fint.copy()

    def test_internal_force_layout_matches_independent_thermal_assembly(self):
        element = self.make_element()
        data = self.make_data()
        internal_force = self.evaluate_internal_force(element, data)
        d_dofs, t_dofs = element.splitDofIDs(4)

        conduction = np.zeros((4, 4))
        capacity = np.zeros((4, 4))
        for i_data in getElemShapeData(self.coords):
            conduction += (
                i_data.dhdx @ element.D @ i_data.dhdx.transpose() * i_data.weight
            )
            capacity += (
                element.capac * np.outer(i_data.h, i_data.h) * i_data.weight
            )

        thermal_tangent = conduction + capacity / self.dtime
        previous_thermal_tangent = capacity / self.dtime
        temperature = data.state[t_dofs]
        previous_temperature = temperature - data.Dstate[t_dofs]
        expected = np.zeros(12)
        expected[t_dofs] = (
            thermal_tangent @ temperature
            - previous_thermal_tangent @ previous_temperature
        )

        np.testing.assert_allclose(
            internal_force[t_dofs], expected[t_dofs], rtol=1.0e-13, atol=1.0e-13
        )
        self.assertGreater(np.linalg.norm(internal_force[d_dofs]), 0.0)
        self.assertGreater(np.linalg.norm(internal_force[t_dofs]), 0.0)

        tangent_data = self.make_data()
        tangent_element = self.make_element()
        tangent_element.mat.reset()
        tangent_element.getTangentStiffness(tangent_data)
        np.testing.assert_array_equal(internal_force, tangent_data.fint)

    def test_tangent_matches_internal_force_central_differences(self):
        element = self.make_element()
        data = self.make_data()
        element.mat.reset()
        element.getTangentStiffness(data)

        finite_difference = np.zeros((12, 12))
        step = 1.0e-7
        for column in range(12):
            plus = self.make_data()
            minus = self.make_data()
            plus.state[column] += step
            minus.state[column] -= step
            plus.Dstate[column] += step
            minus.Dstate[column] -= step
            force_plus = self.evaluate_internal_force(element, plus)
            force_minus = self.evaluate_internal_force(element, minus)
            finite_difference[:, column] = (force_plus - force_minus) / (2.0 * step)

        d_dofs, t_dofs = element.splitDofIDs(4)
        blocks = [
            ("displacement,displacement", d_dofs, d_dofs),
            ("displacement,temperature", d_dofs, t_dofs),
            ("temperature,displacement", t_dofs, d_dofs),
            ("temperature,temperature", t_dofs, t_dofs),
        ]
        for name, rows, columns in blocks:
            with self.subTest(block=name):
                indices = np.ix_(rows, columns)
                np.testing.assert_allclose(
                    data.stiff[indices],
                    finite_difference[indices],
                    rtol=5.0e-6,
                    atol=5.0e-5,
                )


if __name__ == "__main__":
    unittest.main()
