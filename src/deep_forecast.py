"""Deep learning utilities for forecasting retail product demand."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential, load_model
import joblib

DATA_PATH = Path("data/sales_data.csv")
MODEL_PATH = Path("models/forecast_model.h5")
SCALER_PATH = Path("models/forecast_scaler.pkl")
METADATA_PATH = Path("models/forecast_metadata.json")
FORECAST_OUTPUT = Path("data/forecast_results.csv")
PLOT_OUTPUT = Path("img/forecast_trend.png")


@dataclass
class ForecastResult:
    """Container for forecast artefacts."""

    history: pd.DataFrame
    forecast: pd.DataFrame
    trend: str
    confidence_interval: Tuple[float, float]


def load_sales_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and preprocess the sales data."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Sales data not found at {data_path}. Please provide a CSV file with "
            "columns: date, product_name, quantity."
        )

    df = pd.read_csv(data_path)
    if "date" not in df.columns or "quantity" not in df.columns:
        raise ValueError("sales_data.csv must contain 'date' and 'quantity' columns.")

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)

    grouped = df.groupby("date")["quantity"].sum().reset_index()
    grouped.rename(columns={"quantity": "total_quantity"}, inplace=True)
    return grouped


def create_sequences(series: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create rolling window sequences for time-series modelling."""
    x, y = [], []
    for idx in range(len(series) - window):
        x.append(series[idx : idx + window])
        y.append(series[idx + window])
    X = np.array(x)
    y = np.array(y)
    return X.reshape((X.shape[0], X.shape[1], 1)), y


def build_model(window_size: int) -> Sequential:
    """Construct a simple LSTM-based forecasting model."""
    model = Sequential(
        [
            LSTM(64, input_shape=(window_size, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def train_forecast_model(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
    metadata_path: Path = METADATA_PATH,
    forecast_output: Path = FORECAST_OUTPUT,
    plot_output: Path = PLOT_OUTPUT,
    window_size: int = 14,
    forecast_horizon: int = 14,
    epochs: int = 50,
    batch_size: int = 16,
) -> ForecastResult:
    """Train the LSTM model and generate future demand forecasts."""
    history = load_sales_data(data_path)
    values = history["total_quantity"].values.astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values.reshape(-1, 1))

    X, y = create_sequences(scaled_values, window_size)
    if len(X) == 0:
        raise ValueError(
            "Not enough historical data to train the forecast model. Increase the dataset or reduce the window size."
        )
    model = build_model(window_size)
    callbacks = [EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)]
    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "window_size": window_size,
        "forecast_horizon": forecast_horizon,
        "data_path": data_path.as_posix(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    forecast = generate_forecast(
        model,
        scaler,
        scaled_values,
        history,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
    )

    forecast_output.parent.mkdir(parents=True, exist_ok=True)
    forecast.forecast.to_csv(forecast_output, index=False)

    plot_forecast(forecast.history, forecast.forecast, plot_output)
    return forecast


def generate_forecast(
    model: Sequential,
    scaler: MinMaxScaler,
    scaled_series: np.ndarray,
    history: pd.DataFrame,
    window_size: int,
    forecast_horizon: int,
) -> ForecastResult:
    """Generate iterative predictions for the specified horizon."""
    last_sequence = scaled_series[-window_size:].reshape(1, window_size, 1)
    predictions: List[float] = []

    for _ in range(forecast_horizon):
        pred = model.predict(last_sequence, verbose=0)[0][0]
        predictions.append(pred)
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = pred

    forecast_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    last_date = history["date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_horizon)
    forecast_df = pd.DataFrame({"date": future_dates, "expected_demand": forecast_values})

    trend_info = calculate_trend(history, forecast_df)
    return ForecastResult(history=history, forecast=forecast_df, **trend_info)


def calculate_trend(history: pd.DataFrame, forecast: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """Determine the future trend and confidence interval."""
    last_actual = history["total_quantity"].iloc[-1]
    future_mean = float(forecast["expected_demand"].mean())
    future_std = float(forecast["expected_demand"].std(ddof=0))

    delta = future_mean - last_actual
    threshold = 0.05 * max(last_actual, 1.0)
    if delta > threshold:
        trend = "increase"
    elif delta < -threshold:
        trend = "decrease"
    else:
        trend = "stable"

    confidence_interval = (
        max(future_mean - future_std, 0.0),
        future_mean + future_std,
    )
    return {"trend": trend, "confidence_interval": confidence_interval}


def plot_forecast(history: pd.DataFrame, forecast: pd.DataFrame, output_path: Path) -> None:
    """Save a line chart of historical data and forecasted demand."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["date"], history["total_quantity"], label="Historical demand")
    ax.plot(forecast["date"], forecast["expected_demand"], label="Forecast", linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Quantity")
    ax.set_title("Demand Forecast")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def load_trained_model(
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
    metadata_path: Path = METADATA_PATH,
) -> Tuple[Sequential, MinMaxScaler, Dict[str, int]]:
    """Load the persisted model, scaler and metadata."""
    if not model_path.exists() or not scaler_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Forecast model or metadata missing. Run train_forecast_model first."
        )
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    metadata = json.loads(metadata_path.read_text())
    return model, scaler, metadata


def predict_trend(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
    metadata_path: Path = METADATA_PATH,
) -> ForecastResult:
    """Load the trained model and compute the forecast/trend summary."""
    model, scaler, metadata = load_trained_model(model_path, scaler_path, metadata_path)
    history = load_sales_data(data_path)
    values = history["total_quantity"].values.astype(float)
    window_size = int(metadata.get("window_size", 14))
    if len(values) < window_size:
        raise ValueError(
            "Not enough data points to generate a forecast. Collect more sales data or reduce the window size."
        )
    scaled_values = scaler.transform(values.reshape(-1, 1))

    forecast = generate_forecast(
        model,
        scaler,
        scaled_values,
        history,
        window_size=window_size,
        forecast_horizon=int(metadata.get("forecast_horizon", 14)),
    )
    return forecast


def cli() -> None:
    """Train the forecast model from the command line."""
    parser = argparse.ArgumentParser(description="Train and evaluate the demand forecast model")
    parser.add_argument("--data", type=str, default=DATA_PATH.as_posix(), help="Path to sales CSV")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--horizon", type=int, default=14, help="Forecast horizon in days")
    parser.add_argument("--window", type=int, default=14, help="Input sequence window size")
    args = parser.parse_args()

    train_forecast_model(
        data_path=Path(args.data),
        window_size=args.window,
        forecast_horizon=args.horizon,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    cli()
