"""Deep learning based demand trend forecasting."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import load_model

FORECAST_MODEL_PATH = Path("models/forecast_model.h5")
FORECAST_PLOT_PATH = Path("img/forecast_trend.png")


@dataclass
class TrendForecastResult:
    """Container object returned by :func:`predict_trend`."""

    dates: pd.DatetimeIndex
    history: pd.Series
    prediction_sequence: np.ndarray
    next_value: float
    trend: str
    confidence_interval: Tuple[float, float]


class DeepForecastModel:
    """Forecasting model powered by an LSTM network."""

    def __init__(self, data_path: Path = Path("data/sales_data.csv"), window: int = 5) -> None:
        self.data_path = data_path
        self.window = window
        self.model: Sequential | None = None

    def load_sales_series(self) -> pd.Series:
        """Load and aggregate sales data into a daily time-series."""

        dataframe = pd.read_csv(self.data_path, parse_dates=["date"])
        if dataframe.empty:
            raise ValueError("Sales dataset is empty. Provide data/sales_data.csv with sales records.")
        aggregated = (
            dataframe.groupby("date")["sales"].sum().sort_index()
        )
        return aggregated

    @staticmethod
    def _build_model(input_shape: Tuple[int, int]) -> Sequential:
        model = Sequential()
        model.add(LSTM(64, activation="tanh", input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        return model

    def _create_sequences(self, series: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
        values = np.array(list(series), dtype=np.float32)
        if len(values) <= self.window:
            raise ValueError("Not enough data to create sequences for training.")
        x, y = [], []
        for index in range(len(values) - self.window):
            x.append(values[index : index + self.window])
            y.append(values[index + self.window])
        features = np.array(x)
        labels = np.array(y)
        features = features.reshape((features.shape[0], features.shape[1], 1))
        return features, labels

    def fit(self, epochs: int = 50, batch_size: int = 8) -> TrendForecastResult:
        """Train the LSTM network and persist the model to disk."""

        series = self.load_sales_series()
        features, labels = self._create_sequences(series)
        self.model = self._build_model((self.window, 1))
        early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
        self.model.fit(features, labels, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stop])
        self.model.save(FORECAST_MODEL_PATH)
        return self._summarize_forecast(series, features, labels)

    def _summarize_forecast(self, series: pd.Series, features: np.ndarray, labels: np.ndarray) -> TrendForecastResult:
        assert self.model is not None, "Model must be trained before summarizing forecast."
        history_dates = series.index
        last_window = series.values[-self.window :].reshape(1, self.window, 1)
        next_value = float(self.model.predict(last_window, verbose=0)[0][0])

        in_sample_predictions = self.model.predict(features, verbose=0).flatten()
        residuals = labels - in_sample_predictions
        std_dev = float(np.std(residuals)) if len(residuals) > 1 else 0.0
        ci_lower = next_value - 1.96 * std_dev
        ci_upper = next_value + 1.96 * std_dev
        last_actual = float(series.values[-1])
        if next_value > last_actual * 1.05:
            trend = "increase"
        elif next_value < last_actual * 0.95:
            trend = "decrease"
        else:
            trend = "stable"

        result = TrendForecastResult(
            dates=history_dates,
            history=series,
            prediction_sequence=in_sample_predictions,
            next_value=next_value,
            trend=trend,
            confidence_interval=(ci_lower, ci_upper),
        )
        self._plot_forecast(result)
        return result

    def _plot_forecast(self, result: TrendForecastResult) -> None:
        FORECAST_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(result.dates[: len(result.prediction_sequence)], result.prediction_sequence, label="Model fit")
        plt.plot(result.dates, result.history.values, label="Historical sales")
        plt.scatter(result.dates[-1] + pd.Timedelta(days=1), result.next_value, color="red", label="Next day forecast")
        plt.title("Sales Trend Forecast")
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FORECAST_PLOT_PATH)
        plt.close()

    def load_trained_model(self) -> None:
        """Load a previously trained model from disk."""

        self.model = load_model(FORECAST_MODEL_PATH)

    def predict_trend(self) -> TrendForecastResult:
        """Predict the next demand value using a trained model."""

        if self.model is None:
            self.load_trained_model()
        series = self.load_sales_series()
        features, labels = self._create_sequences(series)
        return self._summarize_forecast(series, features, labels)


def predict_trend() -> TrendForecastResult:
    """Convenience wrapper returning the latest forecast result."""

    model = DeepForecastModel()
    if FORECAST_MODEL_PATH.exists():
        model.load_trained_model()
        return model.predict_trend()
    return model.fit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the deep learning demand forecaster")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size used for training")
    parser.add_argument("--window", type=int, default=5, help="Sliding window size for sequences")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = DeepForecastModel(window=args.window)
    result = model.fit(epochs=args.epochs, batch_size=args.batch_size)
    print(
        "Trend: {trend}, next value: {value:.2f}, confidence interval: ({low:.2f}, {high:.2f})".format(
            trend=result.trend,
            value=result.next_value,
            low=result.confidence_interval[0],
            high=result.confidence_interval[1],
        )
    )


if __name__ == "__main__":
    main()
