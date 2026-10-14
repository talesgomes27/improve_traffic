import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import pandas as pd
import numpy as np

from src.utils import fahrenheit_to_celsius, inches_to_mm


def process_weather(raw_weather: pd.DataFrame) -> pd.DataFrame:
    """Higieniza dados de clima, converte unidades para o SI e trata valores 'T'."""
    df = raw_weather.copy()

    # Conversão de datas (formato D-M-YYYY)
    parsed_dates = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df["pickup_date"] = parsed_dates.dt.strftime("%Y-%m-%d")

    # Mapeamento do valor 'T' (traço residual) para 0.0
    for col in ["precipitation", "snow fall", "snow depth"]:
        df[col] = df[col].astype(str).str.strip().replace("T", "0.0")
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Conversão de temperaturas para Celsius
    df["temp_max_c"] = fahrenheit_to_celsius(df["maximum temperature"].values)
    df["temp_min_c"] = fahrenheit_to_celsius(df["minimum temperature"].values)
    df["temp_avg_c"] = fahrenheit_to_celsius(df["average temperature"].values)

    # Conversão de precipitação e neve para milímetros
    df["precipitation_mm"] = inches_to_mm(df["precipitation"].values)
    df["snow_fall_mm"] = inches_to_mm(df["snow fall"].values)
    df["snow_depth_mm"] = inches_to_mm(df["snow depth"].values)

    target_cols = [
        "pickup_date",
        "temp_max_c",
        "temp_min_c",
        "temp_avg_c",
        "precipitation_mm",
        "snow_fall_mm",
        "snow_depth_mm",
    ]
    return df[target_cols]


def prepare_train_data(raw_train: pd.DataFrame, clean_weather: pd.DataFrame) -> pd.DataFrame:
    """Aplica filtros de ruído operacional e une com dados meteorológicos."""
    df = raw_train.copy()

    # Saneamento de viagens anômalas (menos de 1 minuto ou mais de 2 horas)
    valid_duration = (df["trip_duration"] >= 60) & (df["trip_duration"] <= 7200)
    df = df[valid_duration].copy()

    # Parsing temporal e chave de join
    pickup_dt = pd.to_datetime(df["pickup_datetime"])
    df["pickup_datetime"] = pickup_dt
    df["pickup_date"] = pickup_dt.dt.strftime("%Y-%m-%d")

    # Merge à esquerda com clima
    merged = df.merge(clean_weather, on="pickup_date", how="left")
    return merged


def prepare_test_data(raw_test: pd.DataFrame, clean_weather: pd.DataFrame) -> pd.DataFrame:
    """Prepara o conjunto de teste preservando todas as linhas para inferência."""
    df = raw_test.copy()

    pickup_dt = pd.to_datetime(df["pickup_datetime"])
    df["pickup_datetime"] = pickup_dt
    df["pickup_date"] = pickup_dt.dt.strftime("%Y-%m-%d")

    merged = df.merge(clean_weather, on="pickup_date", how="left")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Ingestão e saneamento dos dados brutos.")
    parser.add_argument("--train", type=str, default="data/train.csv", help="Caminho do train.csv")
    parser.add_argument("--test", type=str, default="data/test.csv", help="Caminho do test.csv")
    parser.add_argument("--weather", type=str, default="data/weather.csv", help="Caminho do weather.csv")
    parser.add_argument("--out-dir", type=str, default="data/processed/", help="Diretório de saída")
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Lendo dados brutos de clima: {args.weather}")
    raw_weather = pd.read_csv(args.weather)
    clean_weather = process_weather(raw_weather)

    print(f"Lendo treino: {args.train}")
    raw_train = pd.read_csv(args.train)
    clean_train = prepare_train_data(raw_train, clean_weather)
    print(f"Treino limpo: {len(clean_train)} registros (removidos {len(raw_train) - len(clean_train)} outliers)")

    print(f"Lendo teste: {args.test}")
    raw_test = pd.read_csv(args.test)
    clean_test = prepare_test_data(raw_test, clean_weather)
    print(f"Teste limpo: {len(clean_test)} registros (preservação de 100%)")

    train_out = out_path / "clean_train.parquet"
    test_out = out_path / "clean_test.parquet"

    clean_train.to_parquet(train_out, index=False)
    clean_test.to_parquet(test_out, index=False)
    print(f"Artefatos salvos com sucesso em {train_out} e {test_out}")


if __name__ == "__main__":
    main()
