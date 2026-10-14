import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.utils import rmsle_score


def split_temporal_data(
    df: pd.DataFrame,
    cutoff_date: str = "2016-06-16",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide os dados em treino e validação com base em uma data de corte estrita."""
    pickup_dt = pd.to_datetime(df["pickup_datetime"])
    mask_train = pickup_dt < cutoff_date
    train_df = df[mask_train].copy()
    val_df = df[~mask_train].copy()
    return train_df, val_df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Seleciona as colunas preditoras legítimas, excluindo identificadores e leakage."""
    excluded = {
        "id",
        "trip_duration",
        "dropoff_datetime",
        "pickup_datetime",
        "pickup_date",
        "store_and_fwd_flag",
    }
    cols = [c for c in df.columns if c not in excluded]
    return sorted(cols)


from sklearn.metrics import r2_score


def evaluate_predictions(y_true_sec: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    """Calcula RMSLE, RMSE, MAE em minutos e R² a partir de predições na escala logarítmica."""
    y_pred_sec = np.expm1(y_pred_log)
    y_pred_sec = np.clip(y_pred_sec, 0.0, None)

    rmsle = rmsle_score(y_true_sec, y_pred_sec)
    diff = y_true_sec - y_pred_sec
    mae_sec = float(np.mean(np.abs(diff)))
    mae_min = float(mae_sec / 60.0)

    rmse_sec = float(np.sqrt(np.mean(diff ** 2)))
    rmse_min = float(rmse_sec / 60.0)

    r2 = float(r2_score(y_true_sec, y_pred_sec))

    return {
        "rmsle": float(rmsle),
        "rmse_seconds": rmse_sec,
        "rmse_minutes": rmse_min,
        "mae_seconds": mae_sec,
        "mae_minutes": mae_min,
        "r2": r2,
    }


def main():
    parser = argparse.ArgumentParser(description="Treinamento e comparação de modelos de machine learning.")
    parser.add_argument("--in-dir", type=str, default="data/processed/", help="Diretório com features Parquet")
    parser.add_argument("--models-dir", type=str, default="models/", help="Diretório para salvar artefatos")
    parser.add_argument("--eval-date", type=str, default="2016-06-16", help="Data de corte para validação temporal")
    args = parser.parse_args()

    in_path = Path(args.in_dir)
    models_path = Path(args.models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    print(f"Carregando features de treino de {in_path / 'features_train.parquet'}")
    df = pd.read_parquet(in_path / "features_train.parquet")

    train_df, val_df = split_temporal_data(df, cutoff_date=args.eval_date)
    print(f"Divisão temporal (corte em {args.eval_date}):")
    print(f" - Treino: {len(train_df)} amostras")
    print(f" - Validação de Junho: {len(val_df)} amostras")

    feature_cols = get_feature_columns(df)
    print(f"Features selecionadas ({len(feature_cols)}): {feature_cols}")

    X_train = train_df[feature_cols].values
    y_train = np.log1p(train_df["trip_duration"].values)

    X_val = val_df[feature_cols].values
    y_val_raw = val_df["trip_duration"].values

    # Definição dos modelos candidatos
    candidates = {
        "Ridge_Baseline": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=100.0, random_state=42)),
        ]),
        "Random_Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=14,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": LGBMRegressor(
            importance_type="gain",
            n_estimators=350,
            learning_rate=0.05,
            num_leaves=45,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        ),
    }

    metrics_results = {}
    best_score = float("inf")
    best_model_name = None
    best_model_obj = None

    for name, model in candidates.items():
        print(f"\nTreinando {name}...")
        model.fit(X_train, y_train)

        preds_log = model.predict(X_val)
        scores = evaluate_predictions(y_val_raw, preds_log)
        metrics_results[name] = scores

        print(
            f" -> {name} - RMSLE: {scores['rmsle']:.4f} | "
            f"RMSE: {scores['rmse_minutes']:.2f} min | "
            f"MAE: {scores['mae_minutes']:.2f} min | "
            f"R²: {scores['r2']:.4f}"
        )

        if scores["rmsle"] < best_score:
            best_score = scores["rmsle"]
            best_model_name = name
            best_model_obj = model

    print(f"\nModelo Campeão: {best_model_name} (RMSLE: {best_score:.4f})")

    # Salva resumo de métricas
    summary_file = models_path / "metrics_summary.json"
    with open(summary_file, "w") as f:
        json.dump(metrics_results, f, indent=2)
    print(f"Métricas salvas em {summary_file}")

    # Salva o melhor modelo treinado
    bundle = {
        "model_name": best_model_name,
        "model": best_model_obj,
        "feature_names": feature_cols,
        "cutoff_date": args.eval_date,
        "val_metrics": metrics_results[best_model_name],
        "all_metrics": metrics_results,
    }
    model_file = models_path / "best_model.joblib"
    joblib.dump(bundle, model_file)
    print(f"Modelo serializado com sucesso em {model_file}")


if __name__ == "__main__":
    main()
