"""Testes de integridade e contratos para o módulo src/data_prep.py."""

import unittest
import pandas as pd
import numpy as np
from src.data_prep import process_weather, prepare_train_data, prepare_test_data


class TestDataPrep(unittest.TestCase):
    def setUp(self):
        # Clima sintético simulando dados brutos do weather.csv
        self.raw_weather = pd.DataFrame({
            "date": ["14-2-2016", "23-1-2016", "1-7-2016"],
            "maximum temperature": [15, 27, 85],
            "minimum temperature": [-1, 24, 68],
            "average temperature": [7.0, 25.5, 76.5],
            "precipitation": ["0.00", "2.31", "T"],
            "snow fall": ["0.0", "27.3", "0.0"],
            "snow depth": ["0", "6", "T"],
        })

        # Treino sintético simulando train.csv
        self.raw_train = pd.DataFrame({
            "id": ["id1", "id2", "id3", "id4"],
            "vendor_id": [1, 2, 1, 2],
            "pickup_datetime": [
                "2016-02-14 10:00:00",
                "2016-01-23 15:30:00",
                "2016-07-01 12:00:00",
                "2016-07-01 18:00:00",
            ],
            "dropoff_datetime": [
                "2016-02-14 10:10:00",
                "2016-01-23 15:45:00",
                "2016-07-01 12:00:10",
                "2016-07-02 18:00:00",
            ],
            "passenger_count": [1, 2, 1, 3],
            "pickup_longitude": [-73.98, -73.99, -73.97, -73.96],
            "pickup_latitude": [40.75, 40.76, 40.74, 40.77],
            "dropoff_longitude": [-73.95, -73.97, -73.97, -73.95],
            "dropoff_latitude": [40.78, 40.74, 40.74, 40.78],
            "store_and_fwd_flag": ["N", "N", "Y", "N"],
            "trip_duration": [600, 900, 10, 86400],  # id3 < 60s, id4 > 7200s
        })

        # Teste sintético simulando test.csv
        self.raw_test = pd.DataFrame({
            "id": ["test1", "test2"],
            "vendor_id": [1, 2],
            "pickup_datetime": [
                "2016-02-14 11:00:00",
                "2016-07-01 09:00:00",
            ],
            "passenger_count": [0, 9],  # Valores atípicos
            "pickup_longitude": [-73.98, -73.99],
            "pickup_latitude": [40.75, 40.76],
            "dropoff_longitude": [-73.95, -73.97],
            "dropoff_latitude": [40.78, 40.74],
            "store_and_fwd_flag": ["N", "N"],
        })

    def test_process_weather_units_and_trace(self):
        weather_clean = process_weather(self.raw_weather)

        # Deve converter 'T' para 0.0
        self.assertEqual(weather_clean.loc[weather_clean["pickup_date"] == "2016-07-01", "precipitation_mm"].values[0], 0.0)
        self.assertEqual(weather_clean.loc[weather_clean["pickup_date"] == "2016-07-01", "snow_depth_mm"].values[0], 0.0)

        # Conversão de temperatura do dia 14/02 (-1°F = -18.33°C)
        feb14_min = weather_clean.loc[weather_clean["pickup_date"] == "2016-02-14", "temp_min_c"].values[0]
        self.assertAlmostEqual(feb14_min, -18.33, places=2)

        # Conversão de neve do dia 23/01 (27.3 in = 693.42 mm)
        jan23_snow = weather_clean.loc[weather_clean["pickup_date"] == "2016-01-23", "snow_fall_mm"].values[0]
        self.assertAlmostEqual(jan23_snow, 693.42, places=2)

    def test_prepare_train_filters_outliers(self):
        weather_clean = process_weather(self.raw_weather)
        train_clean = prepare_train_data(self.raw_train, weather_clean)

        # id3 (< 60s) e id4 (> 7200s) devem ser removidos
        self.assertEqual(len(train_clean), 2)
        self.assertListEqual(train_clean["id"].tolist(), ["id1", "id2"])

        # Deve conter as colunas do clima unidas
        self.assertIn("temp_avg_c", train_clean.columns)
        self.assertIn("precipitation_mm", train_clean.columns)
        self.assertEqual(train_clean.isnull().sum().sum(), 0)

    def test_prepare_test_preserves_all_rows(self):
        weather_clean = process_weather(self.raw_weather)
        test_clean = prepare_test_data(self.raw_test, weather_clean)

        # Todas as linhas devem ser mantidas
        self.assertEqual(len(test_clean), 2)

        # Proibição estrita de leakage
        self.assertNotIn("dropoff_datetime", test_clean.columns)
        self.assertNotIn("trip_duration", test_clean.columns)
        self.assertIn("temp_avg_c", test_clean.columns)


if __name__ == "__main__":
    unittest.main()
