"""Testes unitários e de integridade para o módulo src/features.py."""

import unittest
import pandas as pd
import numpy as np
from src.features import engineer_features


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.sample_df = pd.DataFrame({
            "id": ["1", "2"],
            "vendor_id": [1, 2],
            "pickup_datetime": pd.to_datetime(["2016-02-15 08:30:00", "2016-06-19 18:00:00"]),
            "pickup_longitude": [-73.98, -73.78],
            "pickup_latitude": [40.75, 40.64],  # Ponto 2 próximo a JFK
            "dropoff_longitude": [-73.95, -73.98],
            "dropoff_latitude": [40.78, 40.75],
            "passenger_count": [1, 2],
            "store_and_fwd_flag": ["N", "Y"],
            "temp_max_c": [5.0, 25.0],
            "temp_min_c": [0.0, 18.0],
            "temp_avg_c": [2.5, 21.5],
            "precipitation_mm": [0.0, 6.2],
            "snow_fall_mm": [0.0, 0.0],
            "snow_depth_mm": [10.0, 0.0],
        })

    def test_engineer_features_adds_spatial_and_temporal(self):
        feats = engineer_features(self.sample_df)

        # Checagem de novas colunas espaciais
        self.assertIn("dist_haversine_km", feats.columns)
        self.assertIn("dist_manhattan_km", feats.columns)
        self.assertIn("bearing_deg", feats.columns)
        self.assertIn("is_jfk", feats.columns)
        self.assertIn("is_lga", feats.columns)

        # Checagem de colunas temporais
        self.assertIn("pickup_hour", feats.columns)
        self.assertIn("pickup_dayofweek", feats.columns)
        self.assertIn("is_weekend", feats.columns)
        self.assertIn("is_morning_rush", feats.columns)
        self.assertIn("is_evening_rush", feats.columns)

        # Checagem de colunas meteorológicas
        self.assertIn("is_rainy", feats.columns)
        self.assertIn("has_snow_ground", feats.columns)
        self.assertIn("is_severe_weather", feats.columns)

        # Validação de regras pontuais
        # Linha 0: Segunda-feira 08h30 -> is_morning_rush == 1, is_weekend == 0
        self.assertEqual(feats.loc[0, "is_morning_rush"], 1)
        self.assertEqual(feats.loc[0, "is_weekend"], 0)

        # Linha 1: Domingo 18h00 -> is_weekend == 1, is_evening_rush == 0 (apenas dias úteis)
        self.assertEqual(feats.loc[1, "is_weekend"], 1)
        self.assertEqual(feats.loc[1, "is_evening_rush"], 0)
        self.assertEqual(feats.loc[1, "is_rainy"], 1)
        self.assertEqual(feats.loc[1, "is_jfk"], 1)

        # Valores finitos e sem NaNs
        self.assertEqual(feats.isnull().sum().sum(), 0)


if __name__ == "__main__":
    unittest.main()
