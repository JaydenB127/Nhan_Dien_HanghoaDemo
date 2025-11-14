"""Machine learning based inventory optimisation pipeline."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.deep_forecast import ForecastResult, predict_trend
from src.risk_simulation import run_risk_simulation

DETECTION_LOG = Path("data/detection_log.csv")
RECOMMENDATIONS_PATH = Path("data/recommendations.csv")
MODEL_PATH = Path("models/ml_optimizer.pkl")


@dataclass
class OptimizerData:
    """Dataset required for optimisation model training."""

    features: pd.DataFrame
    targets_reorder: pd.Series
    targets_safety: pd.Series


def load_detection_log(path: Path = DETECTION_LOG) -> pd.DataFrame:
    """Read the detection log produced by the real-time identifier."""
    if not path.exists():
        raise FileNotFoundError(
            "Detection log not found. Run the real-time identifier to generate data."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Detection log is empty. Collect detections before training.")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df.dropna(subset=["confidence", "count"], inplace=True)
    return df


def prepare_feature_dataset(
    detection_df: pd.DataFrame,
    forecast: ForecastResult,
    risk_summary: Dict[str, float],
) -> OptimizerData:
    """Combine detection, forecast and risk data into a modelling dataset."""
    detection_df["timestamp"] = pd.to_datetime(detection_df["timestamp"], errors="coerce")
    detection_df.dropna(subset=["timestamp"], inplace=True)

    aggregates = (
        detection_df.groupby("product_name")
        .agg(count=("count", "sum"), avg_confidence=("confidence", "mean"))
        .reset_index()
    )
    if aggregates.empty:
        raise ValueError("Detection log lacks valid entries for optimisation.")
    aggregates["avg_confidence"].fillna(0.0, inplace=True)

    total_count = float(aggregates["count"].sum()) or 1.0
    total_expected = float(forecast.forecast["expected_demand"].sum())
    aggregates["expected_demand"] = (
        aggregates["count"] / total_count
    ) * total_expected

    aggregates["shortage_prob"] = risk_summary.get("shortage_prob", 0.0)
    aggregates["overstock_prob"] = risk_summary.get("overstock_prob", 0.0)

    aggregates["target_reorder"] = aggregates["expected_demand"] * (
        1.0 + aggregates["shortage_prob"] - aggregates["overstock_prob"]
    )
    aggregates["target_safety"] = aggregates["expected_demand"] * (
        0.2 + aggregates["shortage_prob"]
    )

    feature_columns = [
        "count",
        "avg_confidence",
        "expected_demand",
        "shortage_prob",
        "overstock_prob",
    ]
    augmented_rows = []
    reorder_targets: list[float] = []
    safety_targets: list[float] = []
    for _, row in aggregates.iterrows():
        vector = row[feature_columns].to_numpy(dtype=float)
        augmented_rows.append(vector)
        reorder_targets.append(row["target_reorder"])
        safety_targets.append(row["target_safety"])
        for _ in range(30):
            noise = np.random.normal(loc=1.0, scale=0.05, size=vector.shape)
            augmented_rows.append(vector * noise)
            reorder_targets.append(row["target_reorder"])
            safety_targets.append(row["target_safety"])

    augmented_array = np.vstack(augmented_rows)
    augmented_df = pd.DataFrame(augmented_array, columns=feature_columns)
    augmented_df["target_reorder"] = reorder_targets
    augmented_df["target_safety"] = safety_targets

    features = augmented_df[feature_columns]
    return OptimizerData(
        features=features,
        targets_reorder=augmented_df["target_reorder"],
        targets_safety=augmented_df["target_safety"],
    )


def train_optimizer_model(
    forecast: ForecastResult | None = None,
    risk_summary: Dict[str, float] | None = None,
    detection_log_path: Path = DETECTION_LOG,
    model_path: Path = MODEL_PATH,
    recommendations_path: Path = RECOMMENDATIONS_PATH,
) -> pd.DataFrame:
    """Train the optimisation model and generate recommendations."""
    detection_df = load_detection_log(detection_log_path)
    if forecast is None:
        forecast = predict_trend()
    if risk_summary is None:
        try:
            risk_summary = run_risk_simulation(forecast)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

    dataset = prepare_feature_dataset(detection_df, forecast, risk_summary)
    reorder_model = RandomForestRegressor(n_estimators=200, random_state=42)
    safety_model = RandomForestRegressor(n_estimators=200, random_state=21)

    reorder_model.fit(dataset.features, dataset.targets_reorder)
    safety_model.fit(dataset.features, dataset.targets_safety)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"reorder": reorder_model, "safety": safety_model}, model_path)

    base_features = detection_df.groupby("product_name").agg(
        count=("count", "sum"), avg_confidence=("confidence", "mean")
    )
    total_expected = float(forecast.forecast["expected_demand"].sum())
    total_count = float(base_features["count"].sum()) or 1.0

    recommendations = base_features.copy()
    recommendations["avg_confidence"].fillna(0.0, inplace=True)
    recommendations["expected_demand"] = (
        recommendations["count"] / total_count
    ) * total_expected
    recommendations["shortage_prob"] = risk_summary.get("shortage_prob", 0.0)
    recommendations["overstock_prob"] = risk_summary.get("overstock_prob", 0.0)

    feature_matrix = recommendations[
        ["count", "avg_confidence", "expected_demand", "shortage_prob", "overstock_prob"]
    ]
    if feature_matrix.empty:
        raise ValueError("No aggregated detections available to optimise inventory.")
    recommendations["reorder_qty"] = reorder_model.predict(feature_matrix)
    recommendations["safety_stock"] = safety_model.predict(feature_matrix)

    recommendations.reset_index(inplace=True)
    output_columns = [
        "product_name",
        "reorder_qty",
        "safety_stock",
        "shortage_prob",
        "overstock_prob",
    ]
    recommendations = recommendations[output_columns]
    recommendations_path.parent.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(recommendations_path, index=False)
    return recommendations


def cli() -> None:
    """CLI entry point for training the optimisation model."""
    parser = argparse.ArgumentParser(description="Train the ML-based stock optimiser")
    args = parser.parse_args()

    try:
        forecast = predict_trend()
    except FileNotFoundError as exc:
        raise SystemExit(
            "Forecast model not found. Train it with 'python src/deep_forecast.py'."
        ) from exc

    risk_summary = run_risk_simulation(forecast)
    train_optimizer_model(forecast=forecast, risk_summary=risk_summary)


if __name__ == "__main__":
    cli()
