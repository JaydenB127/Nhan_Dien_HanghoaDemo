"""Deep learning based demand forecasting module."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tensorflow import keras

MODELS_DIR = Path("models")
DATA_DIR = Path("data")
FORECAST_MODEL_PATH = MODELS_DIR / "forecast_model.h5"
FORECAST_META_PATH = MODELS_DIR / "forecast_meta.json"
FORECAST_RESULTS_PATH = DATA_DIR / "forecast_results.csv"
SALES_DATA_PATH = DATA_DIR / "sales_data.csv"

WINDOW_SIZE = 7
FORECAST_HORIZON = 7


@dataclass
class ForecastResult:
    """Container for forecast outputs."""

    dates: List[datetime]
    values: List[float]
    residual_std: float


class RetailDemandForecaster:
    """Train and serve LSTM-based demand forecasts."""

    def __init__(self, window_size: int = WINDOW_SIZE, horizon: int = FORECAST_HORIZON):
        self.window_size = window_size
        self.horizon = horizon
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def load_sales_data(self, path: Path = SALES_DATA_PATH) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(
                "Sales data not found. Please provide data/sales_data.csv or generate detection logs."
            )
        df = pd.read_csv(path, parse_dates=["date"])
        if "units_sold" not in df.columns:
            raise ValueError("sales_data.csv must contain a 'units_sold' column")
        return df

    def _prepare_series(self, df: pd.DataFrame) -> pd.Series:
        series = df.groupby("date")["units_sold"].sum().sort_index()
        return series

    def _create_windows(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x, y = [], []
        for idx in range(len(values) - self.window_size):
            x.append(values[idx : idx + self.window_size])
            y.append(values[idx + self.window_size])
        return np.array(x), np.array(y)

    def _build_model(self) -> keras.Model:
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(self.window_size, 1)),
                keras.layers.LSTM(64, return_sequences=False),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        return model

    def train(self) -> ForecastResult:
        df = self.load_sales_data()
        series = self._prepare_series(df)
        values = series.values.astype(np.float32)

        if len(values) <= self.window_size:
            raise ValueError("Not enough data points to train the model")

        mean = float(np.mean(values))
        std = float(np.std(values) or 1.0)
        normalized = (values - mean) / std

        x, y = self._create_windows(normalized)
        x = x[..., np.newaxis]
        y = y[..., np.newaxis]

        model = self._build_model()
        model.fit(x, y, epochs=50, batch_size=16, verbose=0)
        model.save(FORECAST_MODEL_PATH)

        residuals = (model.predict(x, verbose=0).flatten() - y.flatten()) * std
        residual_std = float(np.std(residuals) or 1.0)

        forecast = self._generate_forecast(
            model, normalized, mean, std, series.index[-1], residual_std
        )

        self._persist_metadata(mean, std, residual_std)
        self._save_forecast_results(forecast)
        return forecast

    def _generate_forecast(
        self,
        model: keras.Model,
        normalized_series: np.ndarray,
        mean: float,
        std: float,
        last_date: pd.Timestamp,
        residual_std: float,
    ) -> ForecastResult:
        history = normalized_series.copy()
        predictions: List[float] = []
        current_window = history[-self.window_size :].reshape(1, self.window_size, 1)

        for _ in range(self.horizon):
            pred = model.predict(current_window, verbose=0)[0][0]
            predictions.append(float(pred))
            history = np.append(history, pred)
            current_window = history[-self.window_size :].reshape(1, self.window_size, 1)

        denormalized = [pred * std + mean for pred in predictions]
        future_dates = [last_date + timedelta(days=i + 1) for i in range(self.horizon)]
        return ForecastResult(future_dates, denormalized, residual_std=residual_std)

    def _persist_metadata(self, mean: float, std: float, residual_std: float) -> None:
        metadata = {"mean": mean, "std": std, "residual_std": residual_std, "window_size": self.window_size}
        with FORECAST_META_PATH.open("w") as file:
            json.dump(metadata, file)

    def _save_forecast_results(self, forecast: ForecastResult) -> None:
        data = {
            "date": [date.strftime("%Y-%m-%d") for date in forecast.dates],
            "predicted_units": forecast.values,
        }
        df = pd.DataFrame(data)
        df.to_csv(FORECAST_RESULTS_PATH, index=False)

    def predict_trend(self) -> Tuple[str, Tuple[float, float]]:
        if not FORECAST_RESULTS_PATH.exists():
            raise FileNotFoundError("Forecast results not found. Please train the model first.")
        forecast_df = pd.read_csv(FORECAST_RESULTS_PATH, parse_dates=["date"])
        sales_df = self.load_sales_data()
        recent_actual = sales_df.groupby("date")["units_sold"].sum().sort_index().tail(self.window_size)
        future_mean = float(forecast_df["predicted_units"].mean())
        recent_mean = float(recent_actual.mean())

        change = future_mean - recent_mean
        threshold = max(5.0, 0.05 * recent_mean)
        if change > threshold:
            trend = "increasing"
        elif change < -threshold:
            trend = "decreasing"
        else:
            trend = "stable"

        metadata = self._load_metadata()
        residual_std = metadata.get("residual_std", 1.0)
        confidence_interval = (future_mean - 1.96 * residual_std, future_mean + 1.96 * residual_std)
        return trend, confidence_interval

    def _load_metadata(self) -> Dict[str, float]:
        if not FORECAST_META_PATH.exists():
            return {}
        with FORECAST_META_PATH.open() as file:
            return json.load(file)


def _run_cli(args: argparse.Namespace) -> None:
    forecaster = RetailDemandForecaster()
    if args.command == "train":
        forecast = forecaster.train()
        print("Model trained and forecast saved to", FORECAST_RESULTS_PATH)
        for date, value in zip(forecast.dates, forecast.values):
            print(date.strftime("%Y-%m-%d"), f"-> {value:.2f}")
    elif args.command == "predict":
        trend, interval = forecaster.predict_trend()
        print(f"Trend: {trend} | Confidence interval: {interval}")
    else:
        raise ValueError(f"Unknown command: {args.command}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retail demand forecasting")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Train the LSTM model")
    subparsers.add_parser("predict", help="Predict demand trend")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    _run_cli(cli_args)
