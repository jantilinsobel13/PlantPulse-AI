"""Train and evaluate the demo model using the included synthetic dataset."""
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

ROOT = Path(__file__).parent
df = pd.read_csv(ROOT / "data" / "industrial_sensor_data.csv")
features = [c for c in df.columns if c != "failure_risk_label"]
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df["failure_risk_label"], test_size=0.2,
    random_state=42, stratify=df["failure_risk_label"]
)
model = RandomForestClassifier(n_estimators=140, max_depth=9, random_state=42,
                               class_weight="balanced")
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(f"Holdout accuracy: {accuracy_score(y_test, predictions):.3f}")
print(classification_report(y_test, predictions, zero_division=0))
print("This evaluation is on synthetic educational data only.")
