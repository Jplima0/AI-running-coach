"""
Gerador de plano de treino periodizado.

Recebe:
- distância-alvo da prova
- número de semanas até a prova
- pace de prova previsto (do performance_model) ou informado pelo usuário
- volume/frequência atual do atleta (das features)

E devolve um plano semana a semana com fases (base/build/peak/taper),
volume progressivo, longão e sessões de qualidade (tempo/intervalado),
com paces calculados a partir do pace de prova previsto.

Isso NÃO é um plano genérico de internet — os números de volume inicial,
frequência semanal e paces vêm diretamente do histórico do atleta.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# fatores de pace por categoria de prova, relativos ao pace de prova previsto
# (pace_treino = pace_prova * fator). fatores maiores = mais lento.
PACE_FACTORS = {
    5: dict(easy=1.25, long=1.20, tempo=1.08, interval=0.93),
    10: dict(easy=1.22, long=1.18, tempo=1.07, interval=0.95),
    21.1: dict(easy=1.18, long=1.15, tempo=1.06, interval=0.97),
    42.2: dict(easy=1.15, long=1.10, tempo=1.05, interval=1.00),
}


def _closest_category(target_distance_km: float) -> float:
    return min(PACE_FACTORS.keys(), key=lambda d: abs(d - target_distance_km))


def compute_training_paces(race_pace_min_km: float, target_distance_km: float) -> dict:
    factors = PACE_FACTORS[_closest_category(target_distance_km)]
    return {name: race_pace_min_km * factor for name, factor in factors.items()}


def _format_pace(pace_min_km: float) -> str:
    total_seconds = int(round(pace_min_km * 60))
    m, s = divmod(total_seconds, 60)
    return f"{m}:{s:02d}/km"


@dataclass
class WeekPlan:
    week_number: int
    phase: str
    weekly_km: float
    long_run_km: float
    is_cutback: bool
    sessions: list = field(default_factory=list)


def _phase_for_week(week_number: int, total_weeks: int) -> str:
    base_end = round(total_weeks * 0.40)
    build_end = base_end + round(total_weeks * 0.35)
    peak_end = build_end + round(total_weeks * 0.15)
    if week_number <= base_end:
        return "Base"
    if week_number <= build_end:
        return "Build"
    if week_number <= peak_end:
        return "Peak"
    return "Taper"


def _target_peak_volume(target_distance_km: float, current_weekly_km: float) -> float:
    """Volume semanal alvo no pico do plano, com base em regras de bolso
    por distância de prova, sem exigir um salto absurdo do volume atual."""
    rule_of_thumb = {
        5: 2.5,
        10: 3.0,
        21.1: 3.5,
        42.2: 4.5,
    }
    category = _closest_category(target_distance_km)
    suggested = target_distance_km * rule_of_thumb[category]
    # não deixa o plano pedir um salto maior que ~80% acima do volume atual
    cap = max(current_weekly_km * 1.8, current_weekly_km + 15)
    baseline = max(current_weekly_km, 10.0)
    return float(np.clip(suggested, baseline, cap))


MAX_WEEKLY_GROWTH = 0.10  # regra clássica: não subir volume mais que ~10%/semana


def _weekly_volumes(total_weeks: int, start_km: float, peak_km: float) -> list:
    """Progressão de volume: sobe gradualmente até o pico respeitando um teto
    de crescimento semanal (~10%), com semana de recuo (cutback) a cada 4ª
    semana, e reduz no taper. Se o pico não for alcançável com segurança
    dentro do número de semanas, o plano simplesmente plateau antes dele —
    nunca força um salto perigoso pra "chegar a tempo".
    """
    base_end = round(total_weeks * 0.40)
    build_end = base_end + round(total_weeks * 0.35)
    peak_end = build_end + round(total_weeks * 0.15)
    ramp_weeks = max(peak_end, 1)

    volumes = []
    current = start_km
    last_non_cutback = start_km

    for w in range(1, total_weeks + 1):
        phase = _phase_for_week(w, total_weeks)

        if phase == "Taper":
            weeks_from_race = total_weeks - w + 1
            frac = {1: 0.35, 2: 0.60}.get(weeks_from_race, 0.75)
            volumes.append(last_non_cutback * frac)
            continue

        is_cutback = (w % 4 == 0) and phase != "Base"

        if is_cutback:
            volumes.append(last_non_cutback * 0.75)
            current = last_non_cutback  # próxima semana retoma de onde parou, não do valor reduzido
            continue

        if current < peak_km:
            # quanto falta pra semana de pico, distribuído com crescimento no máximo de 10%/semana
            remaining_weeks = max(ramp_weeks - w + 1, 1)
            needed_ratio = (peak_km / current) ** (1 / remaining_weeks) if current > 0 else 1.0
            growth = min(needed_ratio - 1, MAX_WEEKLY_GROWTH)
            current = current * (1 + max(growth, 0))
            current = min(current, peak_km)

        volumes.append(current)
        last_non_cutback = current

    return volumes


def generate_plan(
    target_distance_km: float,
    total_weeks: int,
    race_pace_min_km: float,
    current_weekly_km: float,
    current_long_run_km: float,
    runs_per_week: int,
    start_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    runs_per_week = int(np.clip(round(runs_per_week), 3, 6))
    peak_km = _target_peak_volume(target_distance_km, current_weekly_km)

    # o volume inicial parte do que o atleta REALMENTE corre hoje, não da
    # distância da prova — pular direto pra um volume "adequado à prova"
    # ignorando o ponto de partida real é receita pra lesão (viola a regra
    # dos ~10% de aumento semanal).
    start_km = max(current_weekly_km, 8.0)
    start_km = min(start_km, peak_km)

    if peak_km > start_km * 2.2:
        print(
            f"[!] Seu volume atual (~{current_weekly_km:.1f} km/semana) está bem "
            f"abaixo do ideal pra essa prova em {total_weeks} semanas. O plano vai "
            "aumentar o volume de forma gradual e segura, mas talvez não alcance o "
            "pico teoricamente ideal a tempo — considere mais semanas se possível."
        )

    volumes = _weekly_volumes(total_weeks, start_km, peak_km)
    paces = compute_training_paces(race_pace_min_km, target_distance_km)

    max_long_run = min(target_distance_km * 1.15, peak_km * 0.42) if target_distance_km < 42 else min(32.0, peak_km * 0.40)
    # mesma lógica pro longão: parte do que o atleta já corre, com um piso
    # pequeno, e nunca mais da metade do volume semanal atual (regra clássica)
    start_long = current_long_run_km if current_long_run_km > 2 else max(start_km * 0.3, 3.0)
    start_long = min(start_long, start_km * 0.5, max_long_run)

    rows = []
    if start_date is None:
        start_date = pd.Timestamp.today().normalize()

    for i, weekly_km in enumerate(volumes, start=1):
        phase = _phase_for_week(i, total_weeks)
        is_cutback = (i % 4 == 0) and phase not in ("Base", "Taper")

        if phase == "Taper":
            weeks_from_race = total_weeks - i + 1
            long_run = {1: target_distance_km * 0.25, 2: target_distance_km * 0.5}.get(
                weeks_from_race, max_long_run * 0.7
            )
        else:
            progress = i / max(round(total_weeks * 0.90), 1)
            long_run = start_long + (max_long_run - start_long) * min(progress, 1.0)
            if is_cutback:
                long_run *= 0.8

        long_run = round(min(long_run, weekly_km * 0.42), 1)
        weekly_km = round(weekly_km, 1)

        sessions = _build_week_sessions(
            phase=phase,
            runs_per_week=runs_per_week,
            weekly_km=weekly_km,
            long_run_km=long_run,
            paces=paces,
            is_race_week=(i == total_weeks),
        )

        week_start = start_date + pd.Timedelta(weeks=i - 1)
        rows.append({
            "semana": i,
            "fase": phase,
            "periodo": f"{week_start.date()} a {(week_start + pd.Timedelta(days=6)).date()}",
            "km_total": weekly_km,
            "longao_km": long_run,
            "cutback": is_cutback,
            "sessoes": sessions,
        })

    return pd.DataFrame(rows)


def _build_week_sessions(
    phase: str,
    runs_per_week: int,
    weekly_km: float,
    long_run_km: float,
    paces: dict,
    is_race_week: bool,
) -> str:
    easy_pace = _format_pace(paces["easy"])
    tempo_pace = _format_pace(paces["tempo"])
    interval_pace = _format_pace(paces["interval"])
    long_pace = _format_pace(paces["long"])

    if is_race_week:
        return (
            f"Seg: descanso | Ter: 20-30min fácil ({easy_pace}) | "
            f"Qua: descanso | Qui: 15-20min fácil c/ 4x100m rápido | "
            f"Sex: descanso | Sáb: descanso ou trote leve 15min | "
            f"Dom: 🏁 PROVA"
        )

    remaining_km = max(weekly_km - long_run_km, 0)
    n_easy_days = max(runs_per_week - 2, 1)  # reserva 1 dia p/ longão + 1 p/ qualidade
    easy_km_each = round(remaining_km / max(n_easy_days, 1), 1) if n_easy_days else 0

    if phase == "Base":
        quality = f"Corrida progressiva fácil→moderado, foco em constância ({easy_pace})"
    elif phase == "Build":
        quality = f"Tempo run: 20-30min contínuo em ritmo de limiar ({tempo_pace})"
    elif phase == "Peak":
        quality = f"Intervalado: 6-8x800m em {interval_pace} c/ 2min trote entre séries"
    else:  # Taper
        quality = f"Tiros curtos p/ manter ritmo: 4-6x200m em {interval_pace}"

    parts = [f"Longão: {long_run_km:.1f}km em {long_pace}", f"Qualidade: {quality}"]
    if easy_km_each > 0:
        parts.append(f"{n_easy_days}x corrida fácil de ~{easy_km_each:.1f}km em {easy_pace}")
    parts.append("1-2 dias de descanso ou cross-training leve")

    return " | ".join(parts)


if __name__ == "__main__":
    from data_processing import load_running_data
    from feature_engineering import build_features
    from performance_model import predict_race_time

    data = load_running_data("../data/raw/activities.csv")
    feats = build_features(data)
    latest = feats.iloc[-1]

    target_distance = 10.0
    pred = predict_race_time(feats, target_distance)

    plan = generate_plan(
        target_distance_km=target_distance,
        total_weeks=12,
        race_pace_min_km=pred.predicted_pace_min_km,
        current_weekly_km=latest["km_last_28d_avg_week"],
        current_long_run_km=latest["longest_run_last_28d"],
        runs_per_week=latest["runs_last_28d"] / 4,
    )

    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 200)
    print(f"Pace de prova previsto: {_format_pace(pred.predicted_pace_min_km)}\n")
    print(plan[["semana", "fase", "periodo", "km_total", "longao_km", "cutback"]].to_string(index=False))
    print("\nDetalhe da semana 1:")
    print(plan.iloc[0]["sessoes"])
    print("\nDetalhe da última semana (prova):")
    print(plan.iloc[-1]["sessoes"])
