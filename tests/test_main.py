import unittest
import numpy as np
from main import get_phi_functions, compute_field_and_derivatives, compute_entropy_sensor, compute_lagrangian

class TestFieldSimulation(unittest.TestCase):
    def test_get_phi_functions_default(self):
        phi_func, dphi_dx_func, dphi_dt_func = get_phi_functions('default')
        self.assertIsNotNone(phi_func)
        self.assertIsNotNone(dphi_dx_func)
        self.assertIsNotNone(dphi_dt_func)
        x, t = 0.0, 0.0
        val = phi_func(x, t)
        self.assertIsInstance(val, float)

    def test_field_and_derivatives_shape(self):
        phi_func, dphi_dx_func, dphi_dt_func = get_phi_functions('default')
        x = np.linspace(-1, 1, 10)
        t = np.linspace(0, 2, 10)
        X, T, phi, dphi_dx, dphi_dt, d2phi_dx2, d2phi_dt2 = compute_field_and_derivatives(
            x, t, phi_func, dphi_dx_func, dphi_dt_func, 'default')
        self.assertEqual(phi.shape, (10, 10))
        self.assertEqual(dphi_dx.shape, (10, 10))
        self.assertEqual(dphi_dt.shape, (10, 10))
        self.assertEqual(d2phi_dx2.shape, (10, 10))
        self.assertEqual(d2phi_dt2.shape, (10, 10))

    def test_entropy_sensor_values(self):
        phi_func, dphi_dx_func, dphi_dt_func = get_phi_functions('default')
        x = np.linspace(-1, 1, 10)
        t = np.linspace(0, 2, 10)
        X, T, phi, dphi_dx, dphi_dt, d2phi_dx2, d2phi_dt2 = compute_field_and_derivatives(
            x, t, phi_func, dphi_dx_func, dphi_dt_func, 'default')
        S = compute_entropy_sensor(dphi_dx, dphi_dt, d2phi_dx2, d2phi_dt2)
        self.assertTrue(np.all(S >= 0))

    def test_lagrangian_output(self):
        phi_func, dphi_dx_func, dphi_dt_func = get_phi_functions('default')
        x = np.linspace(-1, 1, 10)
        t = np.linspace(0, 2, 10)
        X, T, phi, dphi_dx, dphi_dt, d2phi_dx2, d2phi_dt2 = compute_field_and_derivatives(
            x, t, phi_func, dphi_dx_func, dphi_dt_func, 'default')
        S = compute_entropy_sensor(dphi_dx, dphi_dt, d2phi_dx2, d2phi_dt2)
        L, alpha = compute_lagrangian(phi, dphi_dx, dphi_dt, S, X, T, 0.05)
        self.assertEqual(L.shape, (10, 10))
        self.assertEqual(alpha.shape, (10, 10))

if __name__ == '__main__':
    unittest.main()
