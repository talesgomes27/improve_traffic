"""Funções matemáticas puras e utilitários sem estado para o pipeline."""

import numpy as np


def haversine_distance(
    lon1: np.ndarray,
    lat1: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray,
    radius_km: float = 6371.0,
) -> np.ndarray:
    """Calcula a distância do grande círculo (Haversine) em quilômetros."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    # Evita instabilidade numérica em pontos idênticos
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius_km * c


def manhattan_distance(
    lon1: np.ndarray,
    lat1: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray,
    radius_km: float = 6371.0,
) -> np.ndarray:
    """Calcula a distância Manhattan aproximada em malha ortogonal em km."""
    dlat = haversine_distance(lon1, lat1, lon1, lat2, radius_km=radius_km)
    dlon = haversine_distance(lon1, lat1, lon2, lat1, radius_km=radius_km)
    return dlat + dlon


def calculate_bearing(
    lon1: np.ndarray,
    lat1: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray,
) -> np.ndarray:
    """Calcula o ângulo de deslocamento (bearing) em graus [0, 360]."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_lambda = np.radians(lon2 - lon1)

    y = np.sin(delta_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)

    bearing_rad = np.arctan2(y, x)
    bearing_deg = np.degrees(bearing_rad)
    return (bearing_deg + 360.0) % 360.0


def fahrenheit_to_celsius(f_temp: np.ndarray | float) -> np.ndarray:
    """Converte temperatura de Fahrenheit (°F) para Celsius (°C)."""
    return (np.asarray(f_temp, dtype=np.float64) - 32.0) * 5.0 / 9.0


def inches_to_mm(inches: np.ndarray | float) -> np.ndarray:
    """Converte polegadas (inches) para milímetros (mm)."""
    return np.asarray(inches, dtype=np.float64) * 25.4


def rmsle_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula a raiz do erro quadrático médio na escala logarítmica (RMSLE)."""
    y_true_clean = np.clip(np.asarray(y_true, dtype=np.float64), 0.0, None)
    y_pred_clean = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, None)
    log_diff = np.log1p(y_true_clean) - np.log1p(y_pred_clean)
    return float(np.sqrt(np.mean(log_diff ** 2)))
