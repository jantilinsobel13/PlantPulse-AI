# PlantPulse AI — Industrial Maintenance & Process Optimizer

A beginner-friendly, working Streamlit project for an Infosys internship portfolio. It combines a machine-learning failure-risk estimate with explainable maintenance suggestions and simple process-efficiency indicators.

## Features
- Interactive machine sensor inputs
- Random Forest classifier trained on included demo data
- Low / medium / high risk indicator
- Maintenance action recommendations based on sensor thresholds
- Energy-per-unit and illustrative optimization indicators
- Dataset preview, chart, and CSV download

## Tech stack
Python · Streamlit · Pandas · NumPy · scikit-learn

## Run locally (macOS / Windows / Linux)
1. Install Python 3.10 or newer.
2. Unzip this project and open a terminal inside the `Industrial_AI_Maintenance_Optimizer` folder.
3. Create and activate a virtual environment:

   **macOS/Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   **Windows**
   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
5. Start the app:
   ```bash
   streamlit run app.py
   ```
6. The browser will open a local dashboard. Change the sensor values in the sidebar and click **Analyze machine**.

## Train/evaluate the model separately
```bash
python train_model.py
```

## Run the simple dataset test
```bash
python -m pip install pytest
pytest
```

## How it works
1. `data/industrial_sensor_data.csv` contains synthetic example sensor records.
2. `app.py` splits the records into training and test sets and trains a Random Forest model.
3. The model estimates the probability of the demo failure label for the entered values.
4. Rule-based checks generate practical maintenance prompts.
5. A small process panel calculates energy consumed per produced unit and an illustrative potential energy-reduction indicator.

## Important limitations
- All bundled data and labels are **synthetic**, generated for demonstration and learning.
- The displayed test score is only an evaluation against synthetic data and must not be represented as real-world industrial accuracy.
- The "potential energy reduction" is an illustrative comparison with a fixed demo reference, not a guaranteed saving.
- Do not connect this demo to real machinery or use its output for safety-critical decisions without validated data, proper engineering review, and operational testing.

## Suggested GitHub description
AI-powered industrial maintenance demo using a Random Forest model, simulated sensor data, maintenance recommendations, and process-efficiency indicators.

## Suggested repository topics
`predictive-maintenance` `industrial-ai` `machine-learning` `streamlit` `python` `process-optimization`

## License
MIT — see `LICENSE`.
