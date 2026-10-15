import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import pandas as pd
import numpy as np

from src.utils import haversine_distance, manhattan_distance, calculate_bearing

# Constantes de Domínio Geográfico e Operacional
JFK_BBOX = {"lon_min": -73.82, "lon_max": -73.76, "lat_min": 40.63, "lat_max": 40.66}
LGA_BBOX = {"lon_min": -73.89, "lon_max": -73.85, "lat_min": 40.76, "lat_max": 40.79}

# Faixas de Horário de Pico Comercial (Dias Úteis)
MORNING_RUSH_START, MORNING_RUSH_END = 7, 9
EVENING_RUSH_START, EVENING_RUSH_END = 16, 19

# Limiares de Clima Severo
SEVERE_PRECIP_MM = 5.0
SEVERE_SNOW_FALL_MM = 25.0
SEVERE_SNOW_DEPTH_MM = 50.0


def is_within_bbox(
    lons: np.ndarray,
    lats: np.ndarray,
    bbox: dict[str, float],
) -> np.ndarray:
    """Verifica se coordenadas pertencem a um retângulo geográfico delimitado."""
    return (
        (lons >= bbox["lon_min"])
        & (lons <= bbox["lon_max"])
        & (lats >= bbox["lat_min"])
        & (lats <= bbox["lat_max"])
    )


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
    jfk_pickup = is_within_bbox(p_lon, p_lat, JFK_BBOX)
    jfk_dropoff = is_within_bbox(d_lon, d_lat, JFK_BBOX)
    out["is_jfk"] = (jfk_pickup | jfk_dropoff).astype(int)

    lga_pickup = is_within_bbox(p_lon, p_lat, LGA_BBOX)
    lga_dropoff = is_within_bbox(d_lon, d_lat, LGA_BBOX)
    out["is_lga"] = (lga_pickup | lga_dropoff).astype(int)

    # 2. Atributos Temporais e de Tráfego
    pickup_dt = pd.to_datetime(out["pickup_datetime"])
    out["pickup_hour"] = pickup_dt.dt.hour
    out["pickup_dayofweek"] = pickup_dt.dt.dayofweek
    out["pickup_month"] = pickup_dt.dt.month
    out["is_weekend"] = (out["pickup_dayofweek"] >= 5).astype(int)

    # Picos comerciais (apenas dias úteis)
    is_weekday = out["is_weekend"] == 0
    out["is_morning_rush"] = (
        is_weekday & out["pickup_hour"].between(MORNING_RUSH_START, MORNING_RUSH_END)
    ).astype(int)
    out["is_evening_rush"] = (
        is_weekday & out["pickup_hour"].between(EVENING_RUSH_START, EVENING_RUSH_END)
    ).astype(int)

    # 3. Interações Climáticas
    out["is_rainy"] = (out["precipitation_mm"] > 0.0).astype(int)
    out["has_snow_fall"] = (out["snow_fall_mm"] > 0.0).astype(int)
    out["has_snow_ground"] = (out["snow_depth_mm"] > 0.0).astype(int)

    # Clima severo: chuva forte ou neve substancial
    out["is_severe_weather"] = (
        (out["precipitation_mm"] >= SEVERE_PRECIP_MM)
        | (out["snow_fall_mm"] >= SEVERE_SNOW_FALL_MM)
        | (out["snow_depth_mm"] >= SEVERE_SNOW_DEPTH_MM)
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
