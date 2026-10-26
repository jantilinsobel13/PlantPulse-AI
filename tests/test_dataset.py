from pathlib import Path
import pandas as pd

def test_dataset_exists_and_has_expected_columns():
    path = Path(__file__).parents[1] / "data" / "industrial_sensor_data.csv"
    df = pd.read_csv(path)
    expected = {
        "air_temperature_c", "process_temperature_c", "vibration_mm_s",
        "pressure_bar", "energy_kwh", "production_rate_units_hr",
        "hours_since_service", "failure_risk_label"
    }
    assert expected.issubset(df.columns)
    assert len(df) >= 100
    assert set(df["failure_risk_label"].unique()).issubset({0, 1})
