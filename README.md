# AI Retail Intelligence Platform

The AI Retail Intelligence Platform converts shelf-camera detections into actionable insights for store operators. The system combines real-time product identification, deep-learning demand forecasts, Monte Carlo risk simulations, and machine-learning inventory optimisation into a single workflow with a Streamlit dashboard front-end.

## Features

- **Real-time product recognition** powered by YOLOv8 with support for webcam or IP streams and CSV logging of detections.
- **Deep learning demand forecasting** using an LSTM network trained on historical sales data.
- **Inventory risk simulation** based on Monte Carlo sampling that estimates shortage/overstock probabilities.
- **Machine-learning optimiser** that proposes reorder quantities and safety stock levels.
- **Unified Streamlit dashboard** displaying live camera output, forecast trends, risk analytics, and optimisation results.

## Project Structure

```
SHELF-PRODUCT-IDENTIFIER-MAIN/
├── data/
│   ├── detection_log.csv
│   ├── forecast_results.csv
│   ├── recommendations.csv
│   ├── risk_analysis.csv
│   ├── sales_data.csv
│   └── img/
├── img/
│   ├── detections/
│   ├── forecast_trend.png
│   └── risk_distribution.png
├── models/
│   ├── forecast_metadata.json
│   ├── forecast_model.h5
│   ├── forecast_scaler.pkl
│   ├── ml_optimizer.pkl
│   └── yolov8_best.pt
├── notebooks/
│   ├── deep_forecast_training.ipynb
│   ├── montecarlo_simulation.ipynb
│   ├── predict_image_product.ipynb
│   └── stock_optimization.ipynb
├── src/
│   ├── deep_forecast.py
│   ├── img2vec_resnet18.py
│   ├── ml_optimizer.py
│   ├── realtime_identifier.py
│   └── risk_simulation.py
├── dashboard_app.py
├── main.py
└── requirements.txt
```

## Getting Started

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: The deep-learning components rely on TensorFlow and PyTorch. Depending on your GPU/CPU setup you may need to install platform specific builds of these frameworks.

### 3. Prepare data

1. Place your YOLOv8 weights at `models/yolov8_best.pt`.
2. Provide historical sales data at `data/sales_data.csv` with columns: `date`, `product_name`, `quantity`.
3. (Optional) Seed `data/detection_log.csv` by running the real-time identifier to capture detections.

## Usage

### Real-time identification

```bash
python main.py --source 0 --save
```

- Use `--source 0` for the default webcam or provide an IP stream URL such as `http://192.168.x.x:8080/video`.
- Add `--save` to persist annotated frames under `img/detections/`.
- Detections are appended to `data/detection_log.csv` with timestamps, predicted product labels, confidence scores, and per-frame counts.

### Train the demand forecast model

```bash
python src/deep_forecast.py --data data/sales_data.csv --epochs 50 --window 14 --horizon 14
```

This command trains the LSTM model, saves the weights to `models/forecast_model.h5`, and stores the scaler/metadata files required for inference. Forecast outputs are saved to `data/forecast_results.csv`, and a plot is generated at `img/forecast_trend.png`.

### Run the Monte Carlo risk simulation

```bash
python src/risk_simulation.py --simulations 1000
```

The simulation reads the latest forecast, generates 1,000 demand scenarios, estimates shortage and overstock probabilities, and appends the results to `data/risk_analysis.csv`. A histogram visualising demand variability is written to `img/risk_distribution.png`.

### Train the inventory optimiser

```bash
python src/ml_optimizer.py
```

The optimiser consumes the detection log, latest forecast, and most recent risk summary to train Random Forest regressors that output recommended reorder quantities and safety stock levels. Results are saved in `data/recommendations.csv` and the trained model is stored in `models/ml_optimizer.pkl`.

### Launch the dashboard

```bash
streamlit run dashboard_app.py
```

Or launch directly via Python (this command internally proxies to `streamlit run`):

```bash
python dashboard_app.py
```

The dashboard provides four tabs:

1. **🧃 Camera View** – Displays the latest detection log entries and saved frames.
2. **📈 Trend Forecast** – Shows historical sales, future predictions, and trend confidence.
3. **📊 Risk Simulation** – Summarises Monte Carlo results and allows one-click re-runs.
4. **🧩 Optimization** – Presents reorder & safety stock recommendations and lets you regenerate them.

Use the sidebar **Retrain Model** button to execute the full training pipeline (forecast → risk simulation → optimisation) directly from the UI.

## Data Pipelines

1. **Detection** – `src/realtime_identifier.py` uses YOLOv8 to perform real-time inference and log detections.
2. **Forecast** – `src/deep_forecast.py` trains an LSTM network, produces forecasts, and calculates the expected trend.
3. **Risk** – `src/risk_simulation.py` performs Monte Carlo sampling to quantify shortage/overstock risks.
4. **Optimisation** – `src/ml_optimizer.py` trains machine-learning regressors to determine reorder quantities and safety stock levels.

Each module can be executed independently for testing or integrated end-to-end through the dashboard.

## Notebooks

The `notebooks/` directory includes guided workflows for experimentation:

- `deep_forecast_training.ipynb` – Walkthrough of the LSTM forecast model.
- `montecarlo_simulation.ipynb` – Exploratory Monte Carlo analysis notebook.
- `stock_optimization.ipynb` – Prototype of the ML optimisation logic.
- `predict_image_product.ipynb` – Legacy notebook for YOLO-based image predictions.

## Contributing

1. Fork the repository and create a feature branch.
2. Run formatting and linting tools prior to committing (e.g., `black`, `flake8`).
3. Submit a pull request with a clear summary and testing notes.

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.
