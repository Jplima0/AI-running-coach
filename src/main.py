#!/usr/bin/env python3
"""
AI Running Coach — CLI

Uso básico:
    python main.py --distance 10k --weeks 12
    python main.py --distance 21k --race-date 2026-12-20
    python main.py --distance 42k --weeks 16 --goal-time 3:45:00

O que ele faz:
    1. Carrega e limpa seu histórico do Strava (data_processing.py)
    2. Calcula carga de treino, ACWR, volume recente, tendência de pace
       (feature_engineering.py)
    3. Prevê seu tempo/pace realista pra distância-alvo, combinando um
       modelo de ML (regressão de quantil) com Riegel personalizado
       (performance_model.py)
    4. Gera um plano periodizado de treino até a prova
       (plan_generator.py)
    5. Imprime um resumo no terminal e salva o plano completo em
       output/training_plan.csv e output/training_plan.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from data_processing import load_running_data
from feature_engineering import build_features
from performance_model import predict_race_time
from plan_generator import generate_plan, _format_pace

DISTANCE_ALIASES = {
    "5k": 5.0, "5km": 5.0,
    "10k": 10.0, "10km": 10.0,
    "15k": 15.0, "15km": 15.0,
    "21k": 21.1, "21km": 21.1, "half": 21.1, "meia": 21.1, "meia-maratona": 21.1,
    "42k": 42.2, "42km": 42.2, "marathon": 42.2, "maratona": 42.2,
}


def parse_distance(value: str) -> float:
    key = value.strip().lower()
    if key in DISTANCE_ALIASES:
        return DISTANCE_ALIASES[key]
    try:
        return float(key.replace("km", "").replace(",", "."))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Distância inválida: '{value}'. Use algo como 5k, 10k, 21k, 42k ou um número em km."
        )


def parse_goal_time(value: str) -> float:
    """Aceita HH:MM:SS ou MM:SS, devolve minutos totais."""
    parts = value.strip().split(":")
    if len(parts) == 3:
        h, m, s = (int(p) for p in parts)
        return h * 60 + m + s / 60
    if len(parts) == 2:
        m, s = (int(p) for p in parts)
        return m + s / 60
    raise argparse.ArgumentTypeError(
        f"Tempo-meta inválido: '{value}'. Use HH:MM:SS ou MM:SS."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI Running Coach — gera um plano de treino personalizado a partir do seu histórico do Strava."
    )
    parser.add_argument(
        "--distance", type=parse_distance, default=10.0,
        help="Distância da prova-alvo: 5k, 10k, 21k, 42k ou um valor em km (padrão: 10k)."
    )
    parser.add_argument(
        "--weeks", type=int, default=None,
        help="Número de semanas até a prova (padrão: 12, ignorado se --race-date for usado)."
    )
    parser.add_argument(
        "--race-date", type=str, default=None,
        help="Data da prova (AAAA-MM-DD). Se usado, calcula as semanas automaticamente."
    )
    parser.add_argument(
        "--goal-time", type=parse_goal_time, default=None,
        help="Tempo-meta pra prova, em HH:MM:SS ou MM:SS. Se não informado, usa a previsão do modelo."
    )
    parser.add_argument(
        "--data", type=str, default="../data/raw/activities.csv",
        help="Caminho pro CSV exportado do Strava (padrão: ../data/raw/activities.csv)."
    )
    parser.add_argument(
        "--output-dir", type=str, default="../output",
        help="Pasta onde salvar o plano gerado (padrão: ../output)."
    )
    return parser


def resolve_weeks(args) -> int:
    if args.race_date:
        race_date = pd.Timestamp(args.race_date)
        today = pd.Timestamp.today().normalize()
        delta_weeks = (race_date - today).days / 7
        if delta_weeks < 3:
            print(
                f"[!] A data da prova ({args.race_date}) é muito próxima "
                f"({delta_weeks:.1f} semanas). Usando o mínimo de 3 semanas."
            )
        return max(int(round(delta_weeks)), 3)
    return args.weeks or 12


def print_athlete_summary(feats: pd.DataFrame) -> None:
    latest = feats.iloc[-1]
    total_km = feats["distance_km"].sum()
    print("=" * 70)
    print("RESUMO DO SEU HISTÓRICO")
    print("=" * 70)
    print(f"Corridas registradas:        {len(feats)}")
    print(f"Período:                     {feats['date'].min().date()} a {feats['date'].max().date()}")
    print(f"Distância total acumulada:   {total_km:.0f} km")
    print(f"Volume últimas 4 semanas:    {latest['km_last_28d_avg_week']:.1f} km/semana")
    print(f"Maior corrida (28 dias):     {latest['longest_run_last_28d']:.1f} km")
    print(f"Frequência recente:          {latest['runs_last_28d']/4:.1f} corridas/semana")

    acwr = latest["acwr"]
    if pd.notna(acwr):
        if acwr > 1.5:
            risco = "⚠️  ALTO — carga subindo rápido, risco de lesão maior. Considere um plano mais conservador."
        elif acwr < 0.8:
            risco = "baixo — volume recente destoante do seu costume (pode estar destreinando ou voltando de pausa)."
        else:
            risco = "ok — carga aguda e crônica equilibradas."
        print(f"ACWR (agudo/crônico):        {acwr:.2f} ({risco})")
    print()


def print_prediction(pred, goal_time_min) -> None:
    print("=" * 70)
    print("PREVISÃO DE PERFORMANCE")
    print("=" * 70)
    print(f"Distância-alvo:              {pred.target_distance_km:.1f} km")
    print(f"Método:                      {pred.method}")
    print(f"Tempo previsto (nível atual):{pred.pretty_time():>10}  ({pred.pretty_pace()})")
    print(f"  {pred.notes}")
    if goal_time_min is not None:
        goal_pace = goal_time_min / pred.target_distance_km
        diff_min = goal_time_min - pred.predicted_time_min
        comparativo = "mais rápido" if diff_min < 0 else "mais lento"
        print(
            f"\nSeu tempo-meta:              {goal_time_min:.1f} min "
            f"({_format_pace(goal_pace)}) — {abs(diff_min):.1f} min {comparativo} "
            "que a previsão do modelo."
        )
        if diff_min < -5:
            print(
                "[!] Sua meta é bem mais ambiciosa que o previsto pelo seu histórico atual. "
                "O plano abaixo vai te ajudar a evoluir, mas ajuste as expectativas com carinho."
            )
    print()


def print_plan_table(plan: pd.DataFrame) -> None:
    print("=" * 70)
    print("PLANO DE TREINO — VISÃO GERAL")
    print("=" * 70)
    view = plan[["semana", "fase", "periodo", "km_total", "longao_km", "cutback"]].copy()
    view["cutback"] = view["cutback"].map({True: "recuo", False: ""})
    print(view.to_string(index=False))
    print()


def save_plan(plan: pd.DataFrame, output_dir: Path, pred, goal_time_min) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "training_plan.csv"
    txt_path = output_dir / "training_plan.txt"

    plan.to_csv(csv_path, index=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("AI RUNNING COACH — PLANO DE TREINO\n")
        f.write("=" * 70 + "\n")
        f.write(f"Distância-alvo: {pred.target_distance_km:.1f} km\n")
        f.write(f"Tempo previsto: {pred.pretty_time()} ({pred.pretty_pace()})\n")
        if goal_time_min:
            f.write(f"Tempo-meta informado: {goal_time_min:.1f} min\n")
        f.write("\n")
        for _, row in plan.iterrows():
            f.write(f"--- Semana {row['semana']} ({row['fase']}) — {row['periodo']} ---\n")
            f.write(f"Volume total: {row['km_total']} km | Longão: {row['longao_km']} km\n")
            f.write(f"{row['sessoes']}\n\n")

    return csv_path, txt_path


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Erro: arquivo de dados não encontrado em '{data_path}'.")
        sys.exit(1)

    weeks = resolve_weeks(args)

    print(f"\nCarregando histórico de {data_path}...")
    raw = load_running_data(str(data_path))
    feats = build_features(raw)

    if len(feats) < 5:
        print("Erro: poucos dados de corrida encontrados para gerar um plano confiável.")
        sys.exit(1)

    print_athlete_summary(feats)

    pred = predict_race_time(feats, args.distance)
    print_prediction(pred, args.goal_time)

    race_pace = (args.goal_time / args.distance) if args.goal_time else pred.predicted_pace_min_km

    latest = feats.iloc[-1]
    plan = generate_plan(
        target_distance_km=args.distance,
        total_weeks=weeks,
        race_pace_min_km=race_pace,
        current_weekly_km=latest["km_last_28d_avg_week"],
        current_long_run_km=latest["longest_run_last_28d"],
        runs_per_week=latest["runs_last_28d"] / 4,
    )

    print_plan_table(plan)

    csv_path, txt_path = save_plan(plan, Path(args.output_dir), pred, args.goal_time)
    print(f"Plano completo salvo em:\n  {csv_path}\n  {txt_path}\n")
    print("Dica: abra o .txt pra ver o detalhe dia-a-dia de cada semana.")


if __name__ == "__main__":
    main()
