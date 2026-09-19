"""
Modelo de performance do AI Running Coach.

Ideia: em vez de tentar prever seu tempo de prova com um modelo genérico,
aprendemos a partir das SUAS próprias corridas a relação entre distância,
desnível e pace — e usamos isso pra estimar o que você conseguiria fazer
numa distância-alvo, no seu nível de forma atual.

Duas peças:

1. Fórmula de Riegel personalizada (baseline clássico, não-ML)
   T2 = T1 * (D2 / D1) ** k
   Ajustamos k (o expoente de fadiga) por regressão log-log nas SUAS
   corridas recentes, em vez de usar o k=1.06 genérico da literatura.

2. Regressão de quantil (scikit-learn, GradientBoostingRegressor,
   loss="quantile", alpha baixo) treinada em (distância, desnível,
   volume semanal recente) -> pace. Usar um quantil baixo (ex: 0.15)
   faz o modelo aprender a "envoltória" das suas corridas mais fortes
   pra cada combinação de distância/volume — uma estimativa de esforço
   forte (não a corrida fácil média), condicionada à sua carga de
   treino no momento.

Se houver poucos dados, cai para o baseline de Riegel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

MIN_RUNS_FOR_ML = 25


@dataclass
class PerformancePrediction:
    target_distance_km: float
    predicted_time_min: float
    predicted_pace_min_km: float
    method: str
    riegel_exponent: float | None = None
    notes: str = ""

    def pretty_time(self) -> str:
        total_seconds = int(round(self.predicted_time_min * 60))
        h, rem = divmod(total_seconds, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h}h{m:02d}min{s:02d}s"
        return f"{m}min{s:02d}s"

    def pretty_pace(self) -> str:
        total_seconds = int(round(self.predicted_pace_min_km * 60))
        m, s = divmod(total_seconds, 60)
        return f"{m}:{s:02d} /km"


def fit_riegel_exponent(df: pd.DataFrame, lookback_days: int = 365) -> float:
    """Ajusta o expoente de fadiga de Riegel via regressão log-log
    nas corridas recentes do atleta. Retorna 1.06 (valor de referência
    da literatura) se não houver dados suficientes."""
    recent = df[df["date"] >= df["date"].max() - pd.Timedelta(days=lookback_days)]
    recent = recent[(recent["distance_km"] > 0.5) & (recent["moving_time_s"] > 0)]

    if len(recent) < 8 or recent["distance_km"].nunique() < 4:
        return 1.06

    log_d = np.log(recent["distance_km"].values)
    log_t = np.log(recent["moving_time_s"].values)

    # regressão linear simples em espaço log-log: log(t) = log(a) + k*log(d)
    k, log_a = np.polyfit(log_d, log_t, 1)

    # limites de sanidade — fora disso, algo está estranho nos dados
    if not (0.9 <= k <= 1.25):
        return 1.06
    return float(k)


def riegel_predict(
    df: pd.DataFrame, target_distance_km: float, exclude_flagged: bool = True
) -> PerformancePrediction:
    """Baseline: pega a melhor performance recente (menor tempo equivalente
    já normalizado pela distância) e extrapola com Riegel.

    exclude_flagged: ignora corridas marcadas como não-representativas
    (ex: feitas lesionado/doente) via coluna opcional 'exclude_from_model'.
    """
    recent = df[df["date"] >= df["date"].max() - pd.Timedelta(days=180)]
    if len(recent) < 3:
        recent = df

    if exclude_flagged and "exclude_from_model" in recent.columns:
        recent = recent[~recent["exclude_from_model"].fillna(False)]

    k = fit_riegel_exponent(df)

    # normaliza cada corrida pra um "tempo equivalente" na distância-alvo
    # e pega a melhor (menor) — essa é a corrida que melhor representa
    # sua forma atual pra extrapolar
    equiv_time_s = recent["moving_time_s"] * (
        target_distance_km / recent["distance_km"]
    ) ** k
    best_idx = equiv_time_s.idxmin()
    best_time_s = equiv_time_s.loc[best_idx]

    predicted_time_min = best_time_s / 60
    predicted_pace = predicted_time_min / target_distance_km

    return PerformancePrediction(
        target_distance_km=target_distance_km,
        predicted_time_min=predicted_time_min,
        predicted_pace_min_km=predicted_pace,
        method="Riegel personalizado",
        riegel_exponent=k,
        notes=(
            f"Baseado na corrida de {recent.loc[best_idx, 'distance_km']:.1f} km "
            f"em {recent.loc[best_idx, 'date'].date()}, extrapolada com "
            f"expoente de fadiga k={k:.3f} (ajustado ao seu histórico)."
        ),
    )


def ml_quantile_predict(
    df: pd.DataFrame, target_distance_km: float, quantile: float = 0.15
) -> PerformancePrediction | None:
    """Regressão de quantil condicionada a distância, desnível e volume
    semanal recente. Retorna None se não houver dados suficientes."""
    data = df.dropna(subset=["distance_km", "pace_min_km", "km_last_28d_avg_week"]).copy()
    if "exclude_from_model" in data.columns:
        data = data[~data["exclude_from_model"].fillna(False)]
    if len(data) < MIN_RUNS_FOR_ML:
        return None

    features = ["distance_km", "elevation_gain_m", "km_last_28d_avg_week"]
    data["elevation_gain_m"] = data["elevation_gain_m"].fillna(0)
    X = data[features].values
    y = data["pace_min_km"].values

    model = GradientBoostingRegressor(
        loss="quantile",
        alpha=quantile,
        n_estimators=120,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X, y)

    current_weekly_km = data["km_last_28d_avg_week"].iloc[-1]
    x_target = np.array([[target_distance_km, 0.0, current_weekly_km]])
    predicted_pace = float(model.predict(x_target)[0])
    predicted_time_min = predicted_pace * target_distance_km

    return PerformancePrediction(
        target_distance_km=target_distance_km,
        predicted_time_min=predicted_time_min,
        predicted_pace_min_km=predicted_pace,
        method="Regressão de quantil (ML)",
        notes=(
            f"Modelo treinado em {len(data)} corridas suas. Estimativa "
            f"condicionada ao seu volume semanal recente de "
            f"{current_weekly_km:.1f} km/semana (percentil {int(quantile*100)} "
            "de esforço, ou seja, uma corrida forte, não uma prova máxima)."
        ),
    )


def predict_race_time(df: pd.DataFrame, target_distance_km: float) -> PerformancePrediction:
    """Combina os dois métodos: usa ML se houver dados suficientes,
    senão cai para Riegel. Retorna a média dos dois quando ambos existem,
    dando mais peso ao ML."""
    riegel = riegel_predict(df, target_distance_km)
    ml = ml_quantile_predict(df, target_distance_km)

    if ml is None:
        return riegel

    blended_time = 0.6 * ml.predicted_time_min + 0.4 * riegel.predicted_time_min
    blended_pace = blended_time / target_distance_km

    return PerformancePrediction(
        target_distance_km=target_distance_km,
        predicted_time_min=blended_time,
        predicted_pace_min_km=blended_pace,
        method="Combinado (ML + Riegel)",
        riegel_exponent=riegel.riegel_exponent,
        notes=(
            f"Média ponderada: ML='{ml.pretty_time()}' ({ml.notes}) | "
            f"Riegel='{riegel.pretty_time()}' ({riegel.notes})"
        ),
    )


if __name__ == "__main__":
    from data_processing import load_running_data
    from feature_engineering import build_features

    data = load_running_data("../data/raw/activities.csv")
    feats = build_features(data)

    for dist, label in [(5, "5 km"), (10, "10 km"), (21.1, "meia maratona"), (42.2, "maratona")]:
        pred = predict_race_time(feats, dist)
        print(f"\n--- {label} ---")
        print(f"Método: {pred.method}")
        print(f"Tempo previsto: {pred.pretty_time()} ({pred.pretty_pace()})")
        print(pred.notes)
