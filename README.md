# AI Retail Intelligence Platform

The AI Retail Intelligence Platform transforms shelf cameras, sales data, and predictive analytics into a unified workflow that helps retailers monitor products, anticipate demand, quantify inventory risk, and optimize replenishment.

## Key Capabilities

- **Realtime shelf intelligence** – Detects beverages and other goods from webcams or IP camera streams using YOLOv8, draws bounding boxes, counts items, and logs detections to CSV.
- **Deep-learning demand forecasts** – Trains an LSTM model on sales data to predict future consumption and classify the upcoming trend.
- **Monte Carlo risk simulation** – Generates thousands of demand scenarios to estimate stock-out probability, overstock risk, and expected financial impact.
- **Machine-learning optimization** – Learns reorder quantities and safety stock recommendations by combining detections, forecasts, and risk metrics.
- **Streamlit dashboard** – Presents live detections, demand forecasts, risk distribution, and replenishment guidance in a single interface with the ability to retrain the full pipeline.

## Project Structure

```
SHELF-PRODUCT-IDENTIFIER-MAIN/
├── data/
│   ├── detection_log.csv
│   ├── img/
│   ├── recommendations.csv
│   ├── risk_analysis.csv
│   ├── sales_data.csv
│   └── ... (generated files e.g. forecast_results.csv)
├── img/
│   └── detections/
├── models/
│   ├── forecast_model.h5
│   ├── ml_optimizer.pkl
│   ├── ml_optimizer_safety.pkl
│   └── yolov8_best.pt (add your custom model here)
├── notebooks/
│   ├── deep_forecast_training.ipynb
│   ├── montecarlo_simulation.ipynb
│   ├── predict_image_product.ipynb
│   └── stock_optimization.ipynb
├── src/
│   ├── deep_forecast.py
│   ├── img2vec_resnet18.py (legacy helper)
│   ├── ml_optimizer.py
│   ├── realtime_identifier.py
│   └── risk_simulation.py
├── dashboard_app.py
├── main.py
└── requirements.txt
```

## Installation

1. Create and activate a Python 3.10+ virtual environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Download or train a YOLOv8 model and place the weights at `models/yolov8_best.pt`.

> **Tip:** GPU acceleration is recommended for realtime detection and neural network training.

## Realtime Product Identification

Run the detector with a webcam or IP stream:

```bash
python main.py --source 0
# or
python main.py --source http://192.168.x.x:8080/video --save
```

The application displays bounding boxes, average frame confidence, per-product counts, and appends detections to `data/detection_log.csv`. When `--save` is provided, annotated frames are written to `img/detections/`.

## Demand Forecasting

Train the LSTM model on historical sales data:

```bash
python src/deep_forecast.py train
```

The script saves the model to `models/forecast_model.h5`, stores metadata, and writes the next 7-day forecast to `data/forecast_results.csv`. Inspect the predicted trend and confidence interval:

```bash
python src/deep_forecast.py predict
```

Ensure `data/sales_data.csv` contains at least one product history with the columns `date, product_name, units_sold`.

## Inventory Risk Simulation

After generating forecasts, estimate inventory risk with Monte Carlo simulation:

```bash
python src/risk_simulation.py --inventory 800 --runs 1000
```

Results are saved to `data/risk_analysis.csv` and a demand histogram is exported to `img/risk_distribution.png`.

## Inventory Optimization

Combine detections, forecasts, and risk outputs to recommend reorder levels:

```bash
python src/ml_optimizer.py train
```

Recommendations populate `data/recommendations.csv`, and the trained models are stored in `models/ml_optimizer.pkl` and `models/ml_optimizer_safety.pkl`. Re-run the command with `predict` to refresh recommendations without retraining.

## Streamlit Dashboard

Launch the full analytics dashboard:

```bash
streamlit run dashboard_app.py
```

The dashboard includes four tabs:

1. **🧃 Camera View** – Displays detection summaries and the latest saved frame.
2. **📈 Trend Forecast** – Shows demand forecasts and trend classification.
3. **📊 Risk Simulation** – Presents Monte Carlo outcomes and the risk distribution chart.
4. **🧩 Optimization** – Lists reorder and safety stock suggestions.

Use the **Retrain Model** button to trigger the forecasting, simulation, and optimization pipeline directly from the UI.

## Extending the Platform

- Replace `data/sales_data.csv` with multi-product histories to train richer demand models.
- Add additional computer-vision classes by fine-tuning YOLOv8 and updating the model weights.
- Customize the risk simulator by adjusting inventory levels and cost assumptions to match your business context.
- Integrate the generated CSV outputs with ERP or BI systems for automated replenishment workflows.

## Troubleshooting

- **Missing dependencies:** Ensure CUDA-compatible versions of PyTorch/TensorFlow if GPU acceleration is required.
- **Empty dashboards:** Run each pipeline step (detection → forecast → risk → optimization) or use the retrain button to populate data sources.
- **Camera access:** Use IP streams or platform-specific device indices if the default webcam index fails.

For further experimentation, refer to the notebooks in the `notebooks/` directory.
