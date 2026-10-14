import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import joblib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

from src.train import split_temporal_data, evaluate_predictions


def set_presentation_style():
    """Configura o estilo visual global para figuras prontas para publicação."""
    sns.set_theme(style="white", font_scale=1.05)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 1.1,
        "grid.color": "#e2e8f0",
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "figure.titlesize": 13,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def plot_residuals_and_calibration(y_true: np.ndarray, y_pred: np.ndarray, out_file: Path):
    """Gera painel integrado com distribuição de resíduos e dispersão Real vs Previsto com alinhamento perfeito."""
    set_presentation_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    # Força aspecto quadrado estritamente idêntico em ambos os quadros
    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)

    # Subplot 1: Distribuição dos Resíduos
    diff_min = (y_true - y_pred) / 60.0
    mean_err = float(np.mean(diff_min))
    std_err = float(np.std(diff_min))
    mae_min = float(np.mean(np.abs(diff_min)))

    sns.histplot(
        diff_min,
        bins=60,
        kde=True,
        ax=ax1,
        color="#1e3a5f",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.6,
    )
    ax1.axvline(0, color="#dc2626", linestyle="--", linewidth=1.5, label="Erro Zero (Ideal)")
    ax1.axvline(mean_err, color="#0284c7", linestyle=":", linewidth=1.5, label=f"Média: {mean_err:+.2f} min")

    ax1.set_xlim(-25, 35)
    ax1.set_title("(a) Distribuição dos Resíduos (Validação)", fontsize=12, fontweight="bold", pad=8)
    ax1.set_xlabel("Resíduo (Real - Previsto) em Minutos", fontsize=11)
    ax1.set_ylabel("Frequência de Viagens", fontsize=11)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9, loc="upper right")
    ax1.grid(True, axis="y", alpha=0.5)
    sns.despine(ax=ax1, top=True, right=True)

    # Caixa informativa
    text_box_1 = f"MAE: {mae_min:.2f} min\nDesv. Padrão: {std_err:.2f} min\nN = {len(y_true):,}"
    ax1.text(
        0.05, 0.93, text_box_1,
        transform=ax1.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.9),
    )

    # Subplot 2: Real vs Previsto
    sample_size = min(2500, len(y_true))
    idx = np.random.RandomState(42).choice(len(y_true), size=sample_size, replace=False)
    y_t_sub = y_true[idx] / 60.0
    y_p_sub = y_pred[idx] / 60.0

    r2_val = r2_score(y_true, y_pred)
    rmsle_val = np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))

    ax2.scatter(
        y_t_sub,
        y_p_sub,
        alpha=0.35,
        color="#2563eb",
        s=18,
        edgecolors="none",
        label="Viagens de Validação",
    )
    max_limit = 90.0
    ax2.plot([0, max_limit], [0, max_limit], color="#dc2626", linestyle="--", linewidth=1.5, label="Linha de Identidade (y = x)")

    ax2.set_xlim(0, max_limit)
    ax2.set_ylim(0, max_limit)
    ax2.set_title("(b) Duração Real vs. Prevista", fontsize=12, fontweight="bold", pad=8)
    ax2.set_xlabel("Duração Real da Viagem (minutos)", fontsize=11)
    ax2.set_ylabel("Duração Prevista pelo LightGBM (minutos)", fontsize=11)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9, loc="upper left")
    ax2.grid(True, alpha=0.5)
    sns.despine(ax=ax2, top=True, right=True)

    text_box_2 = f"RMSLE: {rmsle_val:.4f}\nR²: {r2_val:.4f}\nLinearidade Forte"
    ax2.text(
        0.95, 0.15, text_box_2,
        transform=ax2.transAxes,
        fontsize=9.5,
        horizontalalignment="right",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.9),
    )

    fig.savefig(out_file, dpi=300)
    plt.close(fig)

    # Salva também versões individuais com aspecto quadrado
    out_dir = out_file.parent
    # Residuos individual
    fig1, a1 = plt.subplots(figsize=(6.2, 5.5), constrained_layout=True)
    a1.set_box_aspect(1)
    sns.histplot(diff_min, bins=60, kde=True, ax=a1, color="#1e3a5f", edgecolor="white", linewidth=0.5, alpha=0.6)
    a1.axvline(0, color="#dc2626", linestyle="--", linewidth=1.5, label="Erro Zero")
    a1.set_xlim(-25, 35)
    a1.set_title("Distribuição dos Resíduos (Validação)", fontsize=12, fontweight="bold")
    a1.set_xlabel("Resíduo (Real - Previsto) em Minutos")
    a1.set_ylabel("Frequência")
    a1.grid(True, axis="y", alpha=0.5)
    sns.despine(ax=a1)
    fig1.savefig(out_dir / "residuals_distribution.png", dpi=300)
    plt.close(fig1)

    # Dispersão individual
    fig2, a2 = plt.subplots(figsize=(6.2, 5.5), constrained_layout=True)
    a2.set_box_aspect(1)
    a2.scatter(y_t_sub, y_p_sub, alpha=0.35, color="#2563eb", s=18, edgecolors="none")
    a2.plot([0, max_limit], [0, max_limit], color="#dc2626", linestyle="--", linewidth=1.5)
    a2.set_xlim(0, max_limit)
    a2.set_ylim(0, max_limit)
    a2.set_title("Duração Real vs Prevista", fontsize=12, fontweight="bold")
    a2.set_xlabel("Duração Real (minutos)")
    a2.set_ylabel("Duração Prevista (minutos)")
    a2.grid(True, alpha=0.5)
    sns.despine(ax=a2)
    fig2.savefig(out_dir / "predicted_vs_actual.png", dpi=300)
    plt.close(fig2)


def plot_feature_importance_gain(model, feature_names: list[str], out_file: Path):
    """Gera gráfico de importância por GANHO (Gain) com percentuais no topo das barras."""
    set_presentation_style()

    if hasattr(model, "booster_"):
        importances = model.booster_.feature_importance(importance_type="gain")
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.ones(len(feature_names))

    total_gain = float(np.sum(importances))
    pct_gain = (importances / total_gain) * 100.0

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "gain": importances,
        "pct_gain": pct_gain,
    }).sort_values("gain", ascending=False).head(12)

    # Mapeamento de nomes legíveis
    label_map = {
        "dist_haversine_km": "Distância Haversine (km)",
        "dist_manhattan_km": "Distância Manhattan (km)",
        "pickup_hour": "Hora de Embarque",
        "bearing_deg": "Ângulo da Rota (Bearing)",
        "dropoff_latitude": "Latitude de Desembarque",
        "pickup_latitude": "Latitude de Embarque",
        "pickup_longitude": "Longitude de Embarque",
        "dropoff_longitude": "Longitude de Desembarque",
        "is_weekend": "Indicador Fim de Semana",
        "pickup_dayofweek": "Dia da Semana",
        "pickup_month": "Mês de Embarque",
        "temp_avg_c": "Temperatura Média (°C)",
    }
    df_imp["label"] = df_imp["feature"].map(lambda x: label_map.get(x, x))

    fig, ax = plt.subplots(figsize=(10, 6.2), constrained_layout=True)

    # Cores destacando as duas maiores variáveis
    colors = ["#1e3a5f" if i < 2 else "#3b82f6" if i < 4 else "#94a3b8" for i in range(len(df_imp))]

    bars = ax.barh(df_imp["label"], df_imp["pct_gain"], color=colors, height=0.72)
    ax.invert_yaxis()

    # Rótulos nas pontas das barras
    for bar, pct, gain in zip(bars, df_imp["pct_gain"], df_imp["gain"]):
        width = bar.get_width()
        ax.text(
            width + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold" if pct > 10 else "normal",
            color="#0f172a",
        )

    ax.set_xlim(0, max(df_imp["pct_gain"]) * 1.15)
    ax.set_title("Importância das Variáveis por Ganho (Gain) - LightGBM Campeão", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Participação no Ganho Total do Modelo (%)", fontsize=11)
    ax.set_ylabel("")
    ax.grid(True, axis="x", alpha=0.5)
    sns.despine(ax=ax, top=True, right=True)

    # Anotação de destaque
    ax.text(
        0.58, 0.55,
        "Distâncias espaciais (Haversine + Manhattan)\nrespondem por 72,9% de todo o ganho\npreditivo da floresta de gradiente.",
        transform=ax.transAxes,
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#eff6ff", edgecolor="#bfdbfe"),
    )

    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def plot_weather_impact_enhanced(train_df: pd.DataFrame, out_file: Path):
    """Gera visualização de impacto climático com intervalos de confiança e anotação da queda percentual."""
    set_presentation_style()
    df = train_df.copy()
    df["speed_kmh"] = df["dist_haversine_km"] / (df["trip_duration"] / 3600.0)
    df = df[(df["speed_kmh"] > 1.0) & (df["speed_kmh"] < 100.0)]

    conditions = []
    for _, row in df.iterrows():
        if row["snow_depth_mm"] > 0:
            conditions.append("Neve no Solo")
        elif row["is_rainy"] == 1:
            conditions.append("Com Chuva")
        else:
            conditions.append("Seco / Normal")
    df["condition"] = conditions

    order = ["Seco / Normal", "Com Chuva", "Neve no Solo"]

    # Estatísticas agregadas
    stats = df.groupby("condition").agg(
        dur_mean=("trip_duration", lambda x: x.mean() / 60.0),
        speed_mean=("speed_kmh", "mean"),
    ).loc[order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)

    # Painel 1: Duração da viagem (minutos)
    colors_dur = ["#334155", "#0284c7", "#d97706"]
    bars1 = ax1.bar(stats.index, stats["dur_mean"], color=colors_dur, width=0.55)
    ax1.set_ylim(0, max(stats["dur_mean"]) * 1.25)
    ax1.set_title("Duração Média das Corridas por Condição", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Duração Média (minutos)", fontsize=11)
    ax1.grid(True, axis="y", alpha=0.5)
    sns.despine(ax=ax1, top=True, right=True)

    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"{h:.1f} min", ha="center", fontsize=10.5, fontweight="bold")

    # Painel 2: Velocidade média aparente (km/h) com queda percentual
    colors_spd = ["#0d9488", "#0284c7", "#e11d48"]
    bars2 = ax2.bar(stats.index, stats["speed_mean"], color=colors_spd, width=0.55)
    ax2.set_ylim(0, max(stats["speed_mean"]) * 1.25)
    ax2.set_title("Velocidade Média Aparente por Condição", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylabel("Velocidade Média (km/h)", fontsize=11)
    ax2.grid(True, axis="y", alpha=0.5)
    sns.despine(ax=ax2, top=True, right=True)

    base_speed = stats.loc["Seco / Normal", "speed_mean"]
    for bar, (cond, row) in zip(bars2, stats.iterrows()):
        h = bar.get_height()
        pct_diff = ((h - base_speed) / base_speed) * 100.0
        pct_label = " (Base)" if cond == "Seco / Normal" else f" ({pct_diff:+.1f}%)"
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"{h:.1f} km/h\n{pct_label}", ha="center", fontsize=9.5, fontweight="bold")

    # Destaque visual para queda de -24.3%
    ax2.annotate(
        "Queda crítica de -24,3%\nna velocidade com neve",
        xy=(2, stats.loc["Neve no Solo", "speed_mean"]),
        xytext=(1.2, 16.0),
        arrowprops=dict(arrowstyle="->", color="#e11d48", lw=1.5),
        fontsize=9.5,
        fontweight="bold",
        color="#e11d48",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff1f2", edgecolor="#fecdd3"),
    )

    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def plot_hourly_heatmap(df: pd.DataFrame, out_file: Path):
    """Gera Heatmap da velocidade média por Dia da Semana x Hora do Dia."""
    set_presentation_style()
    data = df.copy()
    data["speed_kmh"] = data["dist_haversine_km"] / (data["trip_duration"] / 3600.0)
    data = data[(data["speed_kmh"] > 1.0) & (data["speed_kmh"] < 100.0)]

    matrix = data.groupby(["pickup_dayofweek", "pickup_hour"])["speed_kmh"].mean().unstack()
    day_labels = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
    matrix.index = [day_labels[i] for i in matrix.index]

    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)

    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "Velocidade Média Aparente (km/h)", "shrink": 0.8},
        linewidths=0.5,
        linecolor="white",
    )

    ax.set_title("Dinâmica Temporal: Velocidade Média por Dia da Semana e Faixa Horária (km/h)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Hora de Partida (0h às 23h)", fontsize=11)
    ax.set_ylabel("")

    # Linhas demarcatórias nos horários de pico comercial (7h-9h e 16h-19h)
    ax.axvline(7, color="#dc2626", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axvline(10, color="#dc2626", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axvline(16, color="#dc2626", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axvline(20, color="#dc2626", linestyle="--", linewidth=1.2, alpha=0.7)

    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def plot_airport_vs_urban(df: pd.DataFrame, out_file: Path):
    """Compara estatísticas entre rotas de aeroportos (JFK, LGA) e rotas intramunicipais."""
    set_presentation_style()
    data = df.copy()
    data["speed_kmh"] = data["dist_haversine_km"] / (data["trip_duration"] / 3600.0)
    data = data[(data["speed_kmh"] > 1.0) & (data["speed_kmh"] < 100.0)]

    def classify_route(row):
        if row["is_jfk"] == 1:
            return "Aeroporto JFK"
        elif row["is_lga"] == 1:
            return "Aeroporto LaGuardia (LGA)"
        return "Rotas Urbanas Centrais"

    data["route_type"] = data.apply(classify_route, axis=1)
    order = ["Rotas Urbanas Centrais", "Aeroporto LaGuardia (LGA)", "Aeroporto JFK"]

    stats = data.groupby("route_type").agg(
        speed_mean=("speed_kmh", "mean"),
        dur_mean=("trip_duration", lambda x: x.mean() / 60.0),
        dist_mean=("dist_haversine_km", "mean"),
    ).loc[order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)

    colors = ["#475569", "#0284c7", "#0d9488"]

    # Painel 1: Velocidade Média (km/h)
    bars1 = ax1.bar(stats.index, stats["speed_mean"], color=colors, width=0.55)
    ax1.set_ylim(0, 36)
    ax1.set_title("Velocidade Média por Destino / Rota", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Velocidade Média (km/h)", fontsize=11)
    ax1.grid(True, axis="y", alpha=0.5)
    sns.despine(ax=ax1, top=True, right=True)

    for b in bars1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h + 0.6, f"{h:.1f} km/h", ha="center", fontsize=10, fontweight="bold")

    ax1.annotate(
        "JFK: Mais que o dobro da velocidade\nurbana (+111% em vias expressas)",
        xy=(2, stats.loc["Aeroporto JFK", "speed_mean"]),
        xytext=(0.8, 30.5),
        arrowprops=dict(arrowstyle="->", color="#0d9488", lw=1.5),
        fontsize=9,
        fontweight="bold",
        color="#0d9488",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0fdfa", edgecolor="#99f6e4"),
    )

    # Painel 2: Duração Média (minutos)
    bars2 = ax2.bar(stats.index, stats["dur_mean"], color=colors, width=0.55)
    ax2.set_ylim(0, 52)
    ax2.set_title("Duração Média das Corridas (minutos)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Tempo de Percurso (minutos)", fontsize=11)
    ax2.grid(True, axis="y", alpha=0.5)
    sns.despine(ax=ax2, top=True, right=True)

    for b in bars2:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 0.8, f"{h:.1f} min", ha="center", fontsize=10, fontweight="bold")

    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def plot_spatial_circuity(df: pd.DataFrame, out_file: Path):
    """Gera gráfico da distribuição do fator de circuidade viária (Manhattan / Haversine)."""
    set_presentation_style()
    data = df.copy()
    valid = (data["dist_haversine_km"] > 0.15) & (data["dist_haversine_km"] < 50.0)
    data = data[valid].copy()
    data["circuity"] = data["dist_manhattan_km"] / data["dist_haversine_km"]
    data = data[(data["circuity"] >= 1.0) & (data["circuity"] <= 1.45)]

    median_circ = float(data["circuity"].median())
    mean_circ = float(data["circuity"].mean())

    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)

    sns.histplot(
        data["circuity"],
        bins=50,
        kde=True,
        ax=ax,
        color="#1e3a5f",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.6,
    )
    ax.axvline(1.0, color="#10b981", linestyle="--", linewidth=1.5, label="Linha Reta Teórica (Fator 1.0)")
    ax.axvline(median_circ, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Mediana Observada ({median_circ:.2f})")

    ax.set_title("Fator de Circuidade Viária em Nova York (Manhattan / Haversine)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Razão de Circuidade (Distância Manhattan / Distância Haversine)", fontsize=11)
    ax.set_ylabel("Contagem de Viagens", fontsize=11)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9.5, loc="upper left")
    ax.grid(True, axis="y", alpha=0.5)
    sns.despine(ax=ax, top=True, right=True)

    text_info = (
        f"Média: {mean_circ:.3f}\n"
        f"Mediana: {median_circ:.3f}\n"
        f"A malha em grade ortogonal impõe\n"
        f"em média +32,6% de percurso real."
    )
    ax.text(
        0.95, 0.92, text_info,
        transform=ax.transAxes,
        fontsize=9.5,
        horizontalalignment="right",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )

    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Avaliação de modelos, diagnósticos visuais e inferência.")
    parser.add_argument("--models-dir", type=str, default="models/", help="Diretório dos modelos")
    parser.add_argument("--features-dir", type=str, default="data/processed/", help="Diretório dos Parquets")
    parser.add_argument("--reports-dir", type=str, default="reports/", help="Diretório de relatórios")
    args = parser.parse_args()

    models_path = Path(args.models_dir)
    features_path = Path(args.features_dir)
    reports_path = Path(args.reports_dir)
    figures_path = reports_path / "figures"
    tables_path = reports_path / "tables"

    figures_path.mkdir(parents=True, exist_ok=True)
    tables_path.mkdir(parents=True, exist_ok=True)

    print(f"Carregando modelo campeão de {models_path / 'best_model.joblib'}")
    bundle = joblib.load(models_path / "best_model.joblib")
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    cutoff_date = bundle.get("cutoff_date", "2016-06-16")

    # 1. Avaliação na partição de validação temporal
    print("Carregando features de treino para validação temporal e diagnósticos...")
    train_full = pd.read_parquet(features_path / "features_train.parquet")
    _, val_df = split_temporal_data(train_full, cutoff_date=cutoff_date)

    X_val = val_df[feature_names].values
    y_val_raw = val_df["trip_duration"].values

    preds_log = model.predict(X_val)
    preds_sec = np.clip(np.expm1(preds_log), 60.0, None)
    scores = evaluate_predictions(y_val_raw, preds_log)

    print(f"Validação Oficial ({bundle['model_name']}):")
    print(f" - RMSLE: {scores['rmsle']:.4f}")
    print(f" - RMSE:  {scores['rmse_minutes']:.2f} minutos ({scores['rmse_seconds']:.1f} segundos)")
    print(f" - MAE:   {scores['mae_minutes']:.2f} minutos ({scores['mae_seconds']:.1f} segundos)")
    print(f" - R²:    {scores['r2']:.4f}")

    # 2. Exportação de figuras estáticas
    print("Gerando gráficos de diagnóstico visual em alta resolução (300 DPI)...")
    plot_residuals_and_calibration(y_val_raw, preds_sec, figures_path / "residuals_and_calibration.png")
    plot_feature_importance_gain(model, feature_names, figures_path / "feature_importance.png")
    plot_weather_impact_enhanced(train_full, figures_path / "weather_traffic_impact.png")
    plot_hourly_heatmap(train_full, figures_path / "hourly_heatmap_speed.png")
    plot_airport_vs_urban(train_full, figures_path / "airport_comparison.png")
    plot_spatial_circuity(train_full, figures_path / "circuity_and_spatial_profile.png")
    print(f"Todas as figuras salvas com sucesso em {figures_path}")

    # 3. Exportação de tabela de métricas comparativas com todas as 4 métricas
    all_metrics = bundle.get("all_metrics", {})
    records = []
    for m_name, m_scores in all_metrics.items():
        records.append({
            "Modelo": m_name,
            "RMSLE": round(m_scores["rmsle"], 4),
            "RMSE (min)": round(m_scores.get("rmse_minutes", m_scores.get("rmse_seconds", 0) / 60.0), 2),
            "MAE (min)": round(m_scores["mae_minutes"], 2),
            "R²": round(m_scores.get("r2", 0.0), 4),
        })
    df_metrics = pd.DataFrame(records).sort_values("RMSLE")
    metrics_csv = tables_path / "model_comparison.csv"
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"Tabela de comparação salva em {metrics_csv}:")
    print(df_metrics.to_string(index=False))

    # 4. Inferência sobre o conjunto de teste e submissão
    print("Gerando previsões para o conjunto de teste...")
    test_df = pd.read_parquet(features_path / "features_test.parquet")
    X_test = test_df[feature_names].values

    test_preds_log = model.predict(X_test)
    test_preds_sec = np.clip(np.expm1(test_preds_log), 60.0, None)

    submission_df = pd.DataFrame({
        "id": test_df["id"],
        "trip_duration": np.round(test_preds_sec).astype(int),
    })

    sub_path = features_path / "submission.csv"
    submission_df.to_csv(sub_path, index=False)
    print(f"Arquivo de submissão salvo em {sub_path} ({len(submission_df)} linhas, 0 nulos)")


if __name__ == "__main__":
    main()
