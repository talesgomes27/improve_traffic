import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import pandas as pd
import numpy as np

from src.utils import haversine_distance, manhattan_distance, calculate_bearing


def is_within_bbox(
    lons: np.ndarray,
    lats: np.ndarray,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> np.ndarray:
    """Verifica se coordenadas pertencem a um retângulo geográfico delimitado."""
    return (lons >= lon_min) & (lons <= lon_max) & (lats >= lat_min) & (lats <= lat_max)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrai atributos geométricos, temporais e de interação climática."""
    out = df.copy()

    # 1. Atributos Espaciais Vetorizados
    p_lon = out["pickup_longitude"].values
    p_lat = out["pickup_latitude"].values
    d_lon = out["dropoff_longitude"].values
    d_lat = out["dropoff_latitude"].values

    out["dist_haversine_km"] = haversine_distance(p_lon, p_lat, d_lon, d_lat)
    out["dist_manhattan_km"] = manhattan_distance(p_lon, p_lat, d_lon, d_lat)
    out["bearing_deg"] = calculate_bearing(p_lon, p_lat, d_lon, d_lat)

    # Indicadores binários de aeroportos
    # JFK: lat [40.63, 40.66], lon [-73.82, -73.76]
    jfk_pickup = is_within_bbox(p_lon, p_lat, -73.82, -73.76, 40.63, 40.66)
    jfk_dropoff = is_within_bbox(d_lon, d_lat, -73.82, -73.76, 40.63, 40.66)
    out["is_jfk"] = (jfk_pickup | jfk_dropoff).astype(int)

    # LGA: lat [40.76, 40.79], lon [-73.89, -73.85]
    lga_pickup = is_within_bbox(p_lon, p_lat, -73.89, -73.85, 40.76, 40.79)
    lga_dropoff = is_within_bbox(d_lon, d_lat, -73.89, -73.85, 40.76, 40.79)
    out["is_lga"] = (lga_pickup | lga_dropoff).astype(int)

    # 2. Atributos Temporais e de Tráfego
    pickup_dt = pd.to_datetime(out["pickup_datetime"])
    out["pickup_hour"] = pickup_dt.dt.hour
    out["pickup_dayofweek"] = pickup_dt.dt.dayofweek
    out["pickup_month"] = pickup_dt.dt.month
    out["is_weekend"] = (out["pickup_dayofweek"] >= 5).astype(int)

    # Picos comerciais (apenas dias úteis)
    is_weekday = out["is_weekend"] == 0
    out["is_morning_rush"] = (is_weekday & out["pickup_hour"].between(7, 9)).astype(int)
    out["is_evening_rush"] = (is_weekday & out["pickup_hour"].between(16, 19)).astype(int)

    # 3. Interações Climáticas
    out["is_rainy"] = (out["precipitation_mm"] > 0.0).astype(int)
    out["has_snow_fall"] = (out["snow_fall_mm"] > 0.0).astype(int)
    out["has_snow_ground"] = (out["snow_depth_mm"] > 0.0).astype(int)

    # Clima severo: chuva forte (>= 5mm) ou neve substancial
    out["is_severe_weather"] = (
        (out["precipitation_mm"] >= 5.0)
        | (out["snow_fall_mm"] >= 25.0)
        | (out["snow_depth_mm"] >= 50.0)
    ).astype(int)

    # 4. Codificação de Flags
    out["store_and_fwd_num"] = (out["store_and_fwd_flag"] == "Y").astype(int)

    return out


def main():
    parser = argparse.ArgumentParser(description="Engenharia de atributos espaciais, temporais e climáticos.")
    parser.add_argument("--in-dir", type=str, default="data/processed/", help="Diretório de entrada")
    parser.add_argument("--out-dir", type=str, default="data/processed/", help="Diretório de saída")
    args = parser.parse_args()

    in_path = Path(args.in_dir)
    out_path = Path(args.out_dir)

    train_file = in_path / "clean_train.parquet"
    test_file = in_path / "clean_test.parquet"

    print(f"Lendo datasets limpos de {in_path}")
    train_df = pd.read_parquet(train_file)
    test_df = pd.read_parquet(test_file)

    print("Calculando features espaciais, temporais e meteorológicas no treino...")
    feat_train = engineer_features(train_df)
    print(f"Treino com features: {feat_train.shape}")

    print("Calculando features no teste...")
    feat_test = engineer_features(test_df)
    print(f"Teste com features: {feat_test.shape}")

    train_out = out_path / "features_train.parquet"
    test_out = out_path / "features_test.parquet"

    feat_train.to_parquet(train_out, index=False)
    feat_test.to_parquet(test_out, index=False)
    print(f"Features gravadas com sucesso em {train_out} e {test_out}")


if __name__ == "__main__":
    main()
