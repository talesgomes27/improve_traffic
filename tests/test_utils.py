"""Testes unitários para o módulo src/utils.py."""

import unittest
import numpy as np
from src.utils import (
    haversine_distance,
    manhattan_distance,
    calculate_bearing,
    fahrenheit_to_celsius,
    inches_to_mm,
    rmsle_score,
)


class TestUtils(unittest.TestCase):
    def test_haversine_distance_known_points(self):
        # Central Park (lat 40.785091, lon -73.968285) to JFK (lat 40.641311, lon -73.778139)
        lat1 = np.array([40.785091])
        lon1 = np.array([-73.968285])
        lat2 = np.array([40.641311])
        lon2 = np.array([-73.778139])

        dist = haversine_distance(lon1, lat1, lon2, lat2)
        self.assertAlmostEqual(dist[0], 22.64, delta=0.5)

    def test_haversine_distance_zero(self):
        lat = np.array([40.75, 40.80])
        lon = np.array([-73.98, -73.95])
        dist = haversine_distance(lon, lat, lon, lat)
        np.testing.assert_allclose(dist, [0.0, 0.0], atol=1e-5)

    def test_manhattan_distance(self):
        lat1 = np.array([40.70])
        lon1 = np.array([-74.00])
        lat2 = np.array([40.80])
        lon2 = np.array([-73.90])

        h_dist = haversine_distance(lon1, lat1, lon2, lat2)
        m_dist = manhattan_distance(lon1, lat1, lon2, lat2)

        # Distância Manhattan deve ser estritamente maior ou igual à reta Haversine
        self.assertGreaterEqual(m_dist[0], h_dist[0])

    def test_calculate_bearing(self):
        # Rumo ao Norte (mesma longitude, latitude maior)
        b_north = calculate_bearing(
            np.array([-74.0]), np.array([40.0]),
            np.array([-74.0]), np.array([41.0])
        )
        self.assertAlmostEqual(b_north[0], 0.0, delta=1.0)

        # Rumo ao Leste (latitude igual, longitude maior)
        b_east = calculate_bearing(
            np.array([-74.0]), np.array([40.0]),
            np.array([-73.0]), np.array([40.0])
        )
        self.assertAlmostEqual(b_east[0], 90.0, delta=1.5)

    def test_fahrenheit_to_celsius(self):
        f_temps = np.array([32.0, 212.0, -1.0])
        c_temps = fahrenheit_to_celsius(f_temps)
        self.assertAlmostEqual(c_temps[0], 0.0, places=2)
        self.assertAlmostEqual(c_temps[1], 100.0, places=2)
        self.assertAlmostEqual(c_temps[2], -18.33, places=2)

    def test_inches_to_mm(self):
        inches = np.array([0.0, 1.0, 27.3])
        mm = inches_to_mm(inches)
        self.assertAlmostEqual(mm[0], 0.0, places=2)
        self.assertAlmostEqual(mm[1], 25.4, places=2)
        self.assertAlmostEqual(mm[2], 693.42, places=2)

    def test_rmsle_score(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([100.0, 200.0, 300.0])
        self.assertAlmostEqual(rmsle_score(y_true, y_pred), 0.0, places=5)

        y_perturbed = np.array([110.0, 190.0, 310.0])
        score = rmsle_score(y_true, y_perturbed)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 0.1)


if __name__ == "__main__":
    unittest.main()
