import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

INPUT = DATA_DIR / "environmental_data.csv"
RESULTS = DATA_DIR / "environmental_anomaly_results.csv"
MODEL_PATH = MODEL_DIR / "isolation_forest_model.pkl"

df = pd.read_csv(INPUT)

features = [
    "temperature",
    "humidity",
    "co2",
    "pm25",
    "light_intensity",
]

X = df[features]

model = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42,
    n_jobs=-1,
)

model.fit(X)

df["anomaly"] = model.predict(X)
df["anomaly_label"] = df["anomaly"].map({
    1: "Normal",
    -1: "Anomaly",
})

df["anomaly_score"] = model.decision_function(X)

def get_severity(score, label):
    if label == "Normal":
        return "Normal"
    if score < -0.15:
        return "High"
    if score < -0.05:
        return "Medium"
    return "Low"

df["severity"] = [
    get_severity(score, label)
    for score, label in zip(df["anomaly_score"], df["anomaly_label"])
]

df.to_csv(RESULTS, index=False)
joblib.dump(model, MODEL_PATH)

print("Model training completed!")
print("Observations:", len(df))
print("Anomalies:", int((df["anomaly_label"] == "Anomaly").sum()))
print("Anomaly rate:", round((df["anomaly_label"] == "Anomaly").mean() * 100, 2), "%")
print("Results:", RESULTS)
print("Model:", MODEL_PATH)
