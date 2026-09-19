"""
Feature engineering para o AI Running Coach.

Como o export do Strava não trouxe 'training_load' / 'intensity' preenchidos
(86% ausentes), construímos um proxy de carga de treino a partir do que
sempre temos: distância, tempo e pace.

Proxy de carga (heurística tipo TRIMP simplificado):
    load = moving_time_min * intensity_factor
    intensity_factor = (pace_referencia / pace_da_corrida) ** 2

Onde pace_referencia é o pace fácil típico do atleta (mediana móvel dos
últimos 90 dias). Corridas mais rápidas que o normal pesam mais que
corridas fáceis de mesma duração — igual o TRIMP faz com FC.
"""

import pandas as pd
import numpy as np

# Corridas que não representam a forma real do atleta (ex: feitas
# lesionado/doente) e por isso devem ser ignoradas pelos modelos de
# performance, embora continuem no histórico/resumos gerais.
# Adicione o activity_id e um motivo curto aqui quando precisar.
EXCLUDE_FROM_MODEL = {
    14881522479: "maratona corrida lesionado - não representa a forma real",
}


def flag_excluded_runs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["exclude_from_model"] = df["activity_id"].isin(EXCLUDE_FROM_MODEL.keys())
    return df


def add_training_load_proxy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()

    # pace de referência: mediana móvel de 90 dias (janela de tempo, não de linhas)
    df = df.set_index("date")
    reference_pace = (
        df["pace_min_km"].rolling("90D", min_periods=3).median()
    )
    df["reference_pace_min_km"] = reference_pace.values
    df = df.reset_index()

    # fallback pro início da série, quando ainda não há janela suficiente
    df["reference_pace_min_km"] = df["reference_pace_min_km"].fillna(
        df["pace_min_km"].median()
    )

    intensity_factor = (df["reference_pace_min_km"] / df["pace_min_km"]) ** 2
    moving_time_min = df["moving_time_s"] / 60

    df["load_proxy"] = moving_time_min * intensity_factor
    return df


def add_acwr(df: pd.DataFrame) -> pd.DataFrame:
    """Acute:Chronic Workload Ratio — agudo (7d) vs crônico (28d).

    ACWR > ~1.3-1.5 costuma ser associado a maior risco de lesão
    (aumento de carga rápido demais). Usamos isso como sinal de alerta,
    não como verdade absoluta.
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    s = df.set_index("date")["load_proxy"]

    acute = s.rolling("7D", min_periods=1).sum()
    chronic = s.rolling("28D", min_periods=1).sum() / 4  # média semanal equivalente

    df["acute_load_7d"] = acute.values
    df["chronic_load_28d_avg_week"] = chronic.values
    df["acwr"] = (df["acute_load_7d"] / df["chronic_load_28d_avg_week"]).replace(
        [np.inf, -np.inf], np.nan
    )
    return df


def add_rolling_volume(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()
    s = df.set_index("date")["distance_km"]

    df["km_last_7d"] = s.rolling("7D", min_periods=1).sum().values
    df["km_last_28d"] = s.rolling("28D", min_periods=1).sum().values
    df["km_last_28d_avg_week"] = df["km_last_28d"] / 4
    df["runs_last_28d"] = (
        s.rolling("28D", min_periods=1).count().values
    )
    df["longest_run_last_28d"] = (
        s.rolling("28D", min_periods=1).max().values
    )
    return df


def add_pace_trend(df: pd.DataFrame, window_days: int = 90) -> pd.DataFrame:
    """Tendência de pace: regressão linear simples de pace ao longo do
    tempo dentro de uma janela móvel, em segundos/km por semana.
    Negativo = melhorando (ficando mais rápido)."""
    df = df.sort_values("date").reset_index(drop=True).copy()
    dates = df["date"].values.astype("datetime64[D]").astype(float)
    paces = df["pace_sec_km"].values

    trend = np.full(len(df), np.nan)
    for i in range(len(df)):
        cutoff = df["date"].iloc[i] - pd.Timedelta(days=window_days)
        mask = (df["date"] > cutoff) & (df["date"] <= df["date"].iloc[i])
        if mask.sum() >= 4:
            x = dates[mask.values]
            y = paces[mask.values]
            x = x - x.mean()
            if np.any(x != 0):
                slope = np.polyfit(x, y, 1)[0]  # seg/km por dia
                trend[i] = slope * 7  # seg/km por semana

    df["pace_trend_sec_per_week"] = trend
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de feature engineering."""
    df = add_training_load_proxy(df)
    df = add_acwr(df)
    df = add_rolling_volume(df)
    df = add_pace_trend(df)
    df = flag_excluded_runs(df)
    return df


if __name__ == "__main__":
    from data_processing import load_running_data

    data = load_running_data("../data/raw/activities.csv")
    feats = build_features(data)

    print(feats[[
        "date", "distance_km", "pace_min_km", "load_proxy", "acwr",
        "km_last_7d", "km_last_28d_avg_week", "pace_trend_sec_per_week"
    ]].tail(15).to_string(index=False))

    feats.to_csv("../data/processed/running_features.csv", index=False)
