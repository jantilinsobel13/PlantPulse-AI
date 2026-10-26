import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path

st.set_page_config(page_title="PlantPulse AI", page_icon="🏭", layout="wide")
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "industrial_sensor_data.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def train_model():
    data = load_data()
    features = [
        "air_temperature_c", "process_temperature_c", "vibration_mm_s",
        "pressure_bar", "energy_kwh", "production_rate_units_hr",
        "hours_since_service"
    ]
    X, y = data[features], data["failure_risk_label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=140, max_depth=9, random_state=42,
                                   class_weight="balanced")
    model.fit(X_train, y_train)
    score = accuracy_score(y_test, model.predict(X_test))
    return model, features, score

def recommendations(values, risk):
    actions = []
    if values["vibration_mm_s"] >= 6:
        actions.append("Inspect bearings, alignment, and mounting because vibration is elevated.")
    if values["process_temperature_c"] >= 75:
        actions.append("Check cooling flow, lubrication, and temperature sensors.")
    if values["pressure_bar"] >= 8:
        actions.append("Inspect pressure regulator, valves, and filter blockage.")
    if values["energy_kwh"] >= 25:
        actions.append("Review idle time and motor loading; schedule an energy-efficiency check.")
    if values["hours_since_service"] >= 700:
        actions.append("Plan preventive maintenance soon; service hours are high.")
    if values["production_rate_units_hr"] < 60:
        actions.append("Review bottlenecks and setup time before increasing machine speed.")
    if risk >= 0.60:
        actions.insert(0, "Prioritize this machine for inspection before the next production run.")
    elif risk < 0.25:
        actions.append("Continue routine monitoring and follow the standard maintenance schedule.")
    if not actions:
        actions.append("Operating values look within the demo system's normal bands. Keep monitoring.")
    return actions

data = load_data()
model, features, test_accuracy = train_model()

st.title("🏭 PlantPulse AI")
st.caption("AI-Based Industrial Maintenance and Process Optimization System")
st.info("Educational demo using synthetic sensor data. Predictions are not certified for real industrial safety decisions.")

with st.sidebar:
    st.header("Machine Sensor Inputs")
    machine_name = st.selectbox("Machine", ["CNC Machine A", "Conveyor Motor B", "Hydraulic Press C"])
    air = st.slider("Air temperature (°C)", 20, 80, 45)
    process = st.slider("Process temperature (°C)", 25, 100, 60)
    vibration = st.slider("Vibration (mm/s)", 0.2, 12.0, 4.0, 0.1)
    pressure = st.slider("Pressure (bar)", 2.0, 10.0, 6.0, 0.1)
    energy = st.slider("Energy consumption (kWh)", 5.0, 40.0, 19.0, 0.5)
    production = st.slider("Production rate (units/hour)", 30, 120, 80)
    service_hours = st.slider("Hours since last service", 0, 1000, 250, 10)
    predict = st.button("Analyze machine", type="primary", use_container_width=True)

values = {
    "air_temperature_c": air,
    "process_temperature_c": process,
    "vibration_mm_s": vibration,
    "pressure_bar": pressure,
    "energy_kwh": energy,
    "production_rate_units_hr": production,
    "hours_since_service": service_hours
}
input_df = pd.DataFrame([values])[features]
risk = float(model.predict_proba(input_df)[0][1])
risk_pct = round(risk * 100)

st.subheader(f"Machine overview · {machine_name}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted failure risk", f"{risk_pct}%")
c2.metric("Production rate", f"{production} units/hr")
c3.metric("Energy use", f"{energy:.1f} kWh")
c4.metric("Model test accuracy", f"{test_accuracy:.1%}")

if risk >= 0.60:
    st.error("HIGH RISK — schedule an inspection and verify sensor readings.")
elif risk >= 0.30:
    st.warning("MEDIUM RISK — review maintenance recommendations.")
else:
    st.success("LOW RISK — continue routine monitoring.")

left, right = st.columns([1, 1])
with left:
    st.markdown("### Sensor readings")
    sensor_df = pd.DataFrame({
        "Parameter": ["Air temperature (°C)", "Process temperature (°C)", "Vibration (mm/s)",
                      "Pressure (bar)", "Energy (kWh)", "Production (units/hr)"],
        "Current value": [air, process, vibration, pressure, energy, production]
    })
    st.dataframe(sensor_df, hide_index=True, use_container_width=True)
with right:
    st.markdown("### Recommended actions")
    for item in recommendations(values, risk):
        st.markdown(f"- {item}")

st.markdown("---")
st.subheader("Process optimization")
baseline_energy_per_unit = energy / max(production, 1)
target_rate = min(120, production + (5 if risk < 0.30 else 0))
estimated_energy_saving = max(0.0, (energy - 18.0) / max(energy, 1) * 100)
o1, o2, o3 = st.columns(3)
o1.metric("Energy per unit", f"{baseline_energy_per_unit:.3f} kWh")
o2.metric("Suggested production target", f"{target_rate} units/hr")
o3.metric("Potential energy reduction*", f"{estimated_energy_saving:.1f}%")
st.caption("*Illustrative estimate against a demo reference of 18 kWh; validate with real plant data before acting.")

st.markdown("---")
tab1, tab2, tab3 = st.tabs(["Dataset preview", "Failure patterns", "About the model"])
with tab1:
    st.write("Synthetic sensor records used to train the demonstration classifier.")
    st.dataframe(data.head(15), use_container_width=True)
    st.download_button("Download demo dataset (CSV)", data.to_csv(index=False).encode("utf-8"),
                       file_name="industrial_sensor_data.csv", mime="text/csv")
with tab2:
    chart_data = data.groupby(pd.cut(data["vibration_mm_s"], bins=6, include_lowest=True))["failure_risk_label"].mean()
    chart_data.index = chart_data.index.astype(str)
    st.bar_chart(chart_data.rename("Observed demo failure rate"))
    st.caption("Rates reflect the synthetic dataset's generated labels, not real factory failure rates.")
with tab3:
    st.write("A Random Forest classifier learns patterns from seven simulated sensor and operating features.")
    st.write(f"Holdout test accuracy on this synthetic dataset: **{test_accuracy:.1%}**.")
    st.write("The system also applies transparent rule-based checks to suggest maintenance actions and simple process-efficiency indicators.")
    st.write("For a real deployment, replace the synthetic data with validated plant records, evaluate false negatives, and involve qualified maintenance engineers.")
