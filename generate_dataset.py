import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# MicroClimate AI — Synthetic Campus Dataset Generator
# ============================================================

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

locations = {
    "Classroom": {
        "temperature": (27.5, 1.8),
        "humidity": (58, 7),
        "co2": (850, 180),
        "pm25": (22, 7),
        "light_intensity": (430, 90),
    },
    "Computer Lab": {
        "temperature": (28.5, 1.7),
        "humidity": (55, 7),
        "co2": (980, 210),
        "pm25": (24, 7),
        "light_intensity": (620, 100),
    },
    "Library": {
        "temperature": (26.5, 1.4),
        "humidity": (60, 6),
        "co2": (760, 150),
        "pm25": (18, 6),
        "light_intensity": (300, 70),
    },
    "Canteen": {
        "temperature": (30.0, 2.2),
        "humidity": (64, 8),
        "co2": (1100, 260),
        "pm25": (34, 11),
        "light_intensity": (520, 110),
    },
    "Parking Area": {
        "temperature": (32.0, 3.0),
        "humidity": (52, 10),
        "co2": (900, 220),
        "pm25": (48, 16),
        "light_intensity": (760, 150),
    },
    "Garden": {
        "temperature": (25.0, 2.0),
        "humidity": (70, 9),
        "co2": (520, 100),
        "pm25": (14, 5),
        "light_intensity": (680, 170),
    },
}

timestamps = pd.date_range(
    "2026-01-01 08:00:00",
    periods=45 * 48,
    freq="30min",
)

rows = []

for location, profile in locations.items():
    for timestamp in timestamps:
        hour = timestamp.hour + timestamp.minute / 60
        daylight = max(0, np.sin((hour - 6) / 12 * np.pi))
        activity = max(0, np.sin((hour - 8) / 10 * np.pi))

        temp_mean, temp_std = profile["temperature"]
        hum_mean, hum_std = profile["humidity"]
        co2_mean, co2_std = profile["co2"]
        pm_mean, pm_std = profile["pm25"]
        light_mean, light_std = profile["light_intensity"]

        temperature = rng.normal(temp_mean + 1.2 * daylight, temp_std)
        humidity = rng.normal(hum_mean - 3 * daylight, hum_std)
        co2 = rng.normal(co2_mean + 220 * activity, co2_std)
        pm25 = rng.normal(pm_mean + 5 * activity, pm_std)
        light = rng.normal(light_mean * (0.55 + 0.55 * daylight), light_std)

        rows.append([
            timestamp,
            location,
            temperature,
            humidity,
            co2,
            pm25,
            light,
        ])

df = pd.DataFrame(
    rows,
    columns=[
        "timestamp",
        "location",
        "temperature",
        "humidity",
        "co2",
        "pm25",
        "light_intensity",
    ],
)

df["temperature"] = df["temperature"].clip(15, 48)
df["humidity"] = df["humidity"].clip(20, 95)
df["co2"] = df["co2"].clip(300, 3000)
df["pm25"] = df["pm25"].clip(1, 250)
df["light_intensity"] = df["light_intensity"].clip(20, 1200)

# Controlled anomalies for ML demonstration.
anomaly_count = int(len(df) * 0.02)
anomaly_indices = rng.choice(df.index, size=anomaly_count, replace=False)

df.loc[anomaly_indices, "temperature"] += rng.uniform(7, 13, anomaly_count)
df.loc[anomaly_indices, "co2"] += rng.uniform(700, 1600, anomaly_count)
df.loc[anomaly_indices, "pm25"] += rng.uniform(40, 130, anomaly_count)

df["temperature"] = df["temperature"].clip(15, 48)
df["co2"] = df["co2"].clip(300, 4000)
df["pm25"] = df["pm25"].clip(1, 300)

df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

output = DATA_DIR / "environmental_data.csv"
df.to_csv(output, index=False)

print("Dataset generated successfully!")
print("Rows:", len(df))
print("Locations:", df["location"].nunique())
print("Saved to:", output)
