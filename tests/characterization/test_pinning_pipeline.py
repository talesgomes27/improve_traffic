"""Testes de Caracterizacao (Golden Master / Pinning Tests).

Estes testes congelam os comportamentos numericos e estruturais invariantes
do pipeline de dados antes das refatoracoes estruturais do Gate 4.
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from src.data_prep import process_weather, prepare_train_data
from src.features import engineer_features
from src.train import get_feature_columns, evaluate_predictions


class TestPinningPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.root_dir = Path(__file__).resolve().parent.parent.parent
        cls.raw_weather = pd.read_csv(cls.root_dir / "data/weather.csv")
        cls.raw_train = pd.read_csv(cls.root_dir / "data/train.csv").head(50)

    def test_pinning_weather_transformation(self):
        """Congela o schema e invariantes estatisticos de process_weather."""
        clean_w = process_weather(self.raw_weather.head(20))

        # Invariante 1: Dimensao exata (20 amostras, 7 colunas)
        self.assertEqual(clean_w.shape, (20, 7))

        # Invariante 2: Ausencia total de nulos
        self.assertEqual(clean_w.isnull().sum().sum(), 0)

        # Invariante 3: Somas de verificacao congeladas (tolerancia 1e-4)
        np.testing.assert_allclose(clean_w["temp_avg_c"].sum(), 27.5, rtol=1e-4)
        np.testing.assert_allclose(clean_w["precipitation_mm"].sum(), 53.086, rtol=1e-4)

    def test_pinning_train_features_transformation(self):
        """Congela o pipeline completo de dados de treino e engenharia de features."""
        clean_w = process_weather(self.raw_weather)
        clean_tr = prepare_train_data(self.raw_train, clean_w)
        feat_tr = engineer_features(clean_tr)

        # Invariante 1: Dimensoes estritas
        self.assertEqual(clean_tr.shape, (50, 18))
        self.assertEqual(feat_tr.shape, (50, 34))

        # Invariante 2: Zero nulos nas 34 colunas
        self.assertEqual(feat_tr.isnull().sum().sum(), 0)

        # Invariante 3: Conjunto congelado de 28 features preditoras
        feature_cols = get_feature_columns(feat_tr)
        self.assertEqual(len(feature_cols), 28)
        self.assertNotIn("id", feature_cols)
        self.assertNotIn("trip_duration", feature_cols)
        self.assertNotIn("dropoff_datetime", feature_cols)

        # Invariante 4: Somas geometricas e contagens espaciais congeladas
        np.testing.assert_allclose(feat_tr["dist_haversine_km"].sum(), 183.352984, rtol=1e-4)
        np.testing.assert_allclose(feat_tr["dist_manhattan_km"].sum(), 234.207242, rtol=1e-4)
        np.testing.assert_allclose(feat_tr["bearing_deg"].sum(), 6469.555543, rtol=1e-4)
        self.assertEqual(int(feat_tr["is_jfk"].sum()), 1)
        self.assertEqual(int(feat_tr["is_lga"].sum()), 2)

    def test_pinning_evaluation_metrics_invariants(self):
        """Congela a integridade numerica de evaluate_predictions."""
        y_true = np.array([300.0, 600.0, 1200.0, 1800.0])
        # Predicoes na escala log(1 + y) simulando previsoes proximas
        y_pred_log = np.log1p(np.array([310.0, 580.0, 1250.0, 1750.0]))

        scores = evaluate_predictions(y_true, y_pred_log)

        self.assertIn("rmsle", scores)
        self.assertIn("mae_minutes", scores)
        self.assertIn("r2", scores)

        # Congela metricas conhecidas
        self.assertAlmostEqual(scores["rmsle"], 0.0345, places=3)
        self.assertAlmostEqual(scores["mae_minutes"], 0.5417, places=3)
        self.assertGreater(scores["r2"], 0.99)


if __name__ == "__main__":
    unittest.main()
