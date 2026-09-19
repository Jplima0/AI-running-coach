import pandas as pd
import numpy as np
import re

MONTHS = {
    "janv.": 1, "févr.": 2, "mars": 3, "avr.": 4, "mai": 5, "juin": 6,
    "juil.": 7, "août": 8, "sept.": 9, "oct.": 10, "nov.": 11, "déc.": 12
}

def parse_strava_date(value):
    """Convert Strava's French exported date to pandas Timestamp."""
    if pd.isna(value):
        return pd.NaT

    s = str(value).strip()
    match = re.match(
        r"(\d{1,2})\s+([^\s]+)\s+(\d{4}),\s+(\d{2}:\d{2}:\d{2})",
        s
    )
    if not match:
        return pd.NaT

    day, month_text, year, time = match.groups()
    month = MONTHS.get(month_text)
    if month is None:
        return pd.NaT

    return pd.Timestamp(
        year=int(year),
        month=month,
        day=int(day),
        hour=int(time[:2]),
        minute=int(time[3:5]),
        second=int(time[6:8])
    )

def to_number(series):
    """Convert Strava numeric fields, including decimal commas."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

def load_running_data(csv_path="../data/raw/activities.csv"):
    df = pd.read_csv(csv_path)

    # Keep only running activities
    runs = df[df["Type d'activité"].eq("Course à pied")].copy()

    data = pd.DataFrame({
        "activity_id": runs["ID de l'activité"],
        "date": runs["Date de l'activité"].map(parse_strava_date),
        "name": runs["Nom de l'activité"],
        "distance_km": to_number(runs["Distance"]),
        "elapsed_time_s": to_number(runs["Temps écoulé"]),
        "moving_time_s": to_number(runs["Durée de déplacement"]),
        "avg_speed_m_s": to_number(runs["Vitesse moyenne"]),
        "elevation_gain_m": to_number(runs["Dénivelé positif"]),
        "avg_cadence_spm": to_number(runs["Cadence moyenne"]),
        "calories": to_number(runs["Calories"]),
        "training_load": to_number(runs["Charge d’entraînement"]),
        "intensity": to_number(runs["Intensité"]),
    })

    # Pace = moving time / distance
    data["pace_min_km"] = (
        data["moving_time_s"] / 60
    ) / data["distance_km"]

    data["pace_sec_km"] = (
        data["moving_time_s"] / data["distance_km"]
    )

    # Remove invalid activities
    data = data[
        (data["distance_km"] > 0) &
        (data["moving_time_s"] > 0) &
        data["date"].notna()
    ].copy()

    return data.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    data = load_running_data()

    print(f"Corridas encontradas: {len(data)}")
    print(f"Primeira corrida: {data['date'].min()}")
    print(f"Última corrida: {data['date'].max()}")
    print(f"Distância total: {data['distance_km'].sum():.1f} km")

    print("\nÚltimas 10 corridas:")
    print(
        data[
            ["date", "name", "distance_km", "moving_time_s", "pace_min_km"]
        ].tail(10).to_string(index=False)
    )

    data.to_csv("../data/processed/running_data.csv", index=False)
