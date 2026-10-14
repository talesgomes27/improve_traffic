"""Testes para a partição temporal e pipelines de modelagem em src/train.py."""

import unittest
import pandas as pd
import numpy as np
from src.train import split_temporal_data, get_feature_columns


class TestModels(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "id": ["1", "2", "3", "4"],
            "pickup_datetime": pd.to_datetime([
                "2016-05-01 10:00:00",
                "2016-06-15 23:59:59",
                "2016-06-16 00:00:01",
                "2016-06-25 12:00:00",
            ]),
            "dropoff_datetime": pd.to_datetime([
                "2016-05-01 10:10:00",
                "2016-06-16 00:10:00",
                "2016-06-16 00:15:00",
                "2016-06-25 12:20:00",
            ]),
            "trip_duration": [600, 601, 899, 1200],
            "vendor_id": [1, 2, 1, 2],
            "passenger_count": [1, 2, 1, 3],
            "dist_haversine_km": [2.5, 3.1, 1.8, 5.0],
            "dist_manhattan_km": [3.2, 4.0, 2.3, 6.5],
            "bearing_deg": [45.0, 90.0, 180.0, 270.0],
            "pickup_hour": [10, 23, 0, 12],
            "pickup_dayofweek": [6, 2, 3, 5],
            "pickup_month": [5, 6, 6, 6],
            "is_weekend": [1, 0, 0, 1],
            "is_morning_rush": [0, 0, 0, 0],
            "is_evening_rush": [0, 0, 0, 0],
            "temp_avg_c": [15.0, 22.0, 22.5, 25.0],
            "precipitation_mm": [0.0, 0.0, 1.2, 0.0],
            "is_rainy": [0, 0, 1, 0],
            "is_jfk": [0, 0, 0, 1],
            "is_lga": [0, 0, 0, 0],
            "store_and_fwd_num": [0, 0, 0, 1],
        })

    def test_split_temporal_data(self):
        train_df, val_df = split_temporal_data(self.df, cutoff_date="2016-06-16")

        self.assertEqual(len(train_df), 2)
        self.assertEqual(len(val_df), 2)

        self.assertListEqual(train_df["id"].tolist(), ["1", "2"])
        self.assertListEqual(val_df["id"].tolist(), ["3", "4"])

        # Sem vazamento temporal
        self.assertTrue((train_df["pickup_datetime"] < "2016-06-16").all())
        self.assertTrue((val_df["pickup_datetime"] >= "2016-06-16").all())

    def test_get_feature_columns_excludes_leakage(self):
        feats = get_feature_columns(self.df)

        # Proibição absoluta de vazamento de dados
        self.assertNotIn("id", feats)
        self.assertNotIn("trip_duration", feats)
        self.assertNotIn("dropoff_datetime", feats)
        self.assertNotIn("pickup_datetime", feats)
        self.assertNotIn("pickup_date", feats)

        # Inclusão de preditores legítimos
        self.assertIn("dist_haversine_km", feats)
        self.assertIn("dist_manhattan_km", feats)
        self.assertIn("pickup_hour", feats)
        self.assertIn("temp_avg_c", feats)
        self.assertIn("is_jfk", feats)


if __name__ == "__main__":
    unittest.main()
