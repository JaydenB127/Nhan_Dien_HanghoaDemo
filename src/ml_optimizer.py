"""Machine learning optimizer for inventory recommendations."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.risk_simulation import RISK_ANALYSIS_PATH

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
DETECTION_LOG_PATH = DATA_DIR / "detection_log.csv"
RECOMMENDATIONS_PATH = DATA_DIR / "recommendations.csv"
ML_MODEL_PATH = MODELS_DIR / "ml_optimizer.pkl"
SAFETY_MODEL_PATH = MODELS_DIR / "ml_optimizer_safety.pkl"
FORECAST_RESULTS_PATH = DATA_DIR / "forecast_results.csv"


@dataclass
class OptimizationResult:
    recommendations: pd.DataFrame
    model_paths: Tuple[Path, Path]


class InventoryOptimizer:
    """Optimize reorder quantities and safety stock using ML."""

    def __init__(self) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.reorder_model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.safety_model = RandomForestRegressor(n_estimators=200, random_state=42)

    def load_detection_log(self) -> pd.DataFrame:
        if not DETECTION_LOG_PATH.exists():
            raise FileNotFoundError("Detection log not found. Run the realtime identifier first.")
        return pd.read_csv(DETECTION_LOG_PATH)

    def load_forecast(self) -> pd.DataFrame:
        if not FORECAST_RESULTS_PATH.exists():
            raise FileNotFoundError("Forecast results not found. Train the forecasting model first.")
        return pd.read_csv(FORECAST_RESULTS_PATH)

    def load_risk_metrics(self) -> pd.DataFrame:
        if not RISK_ANALYSIS_PATH.exists():
            raise FileNotFoundError("Risk analysis not found. Run the risk simulation first.")
        return pd.read_csv(RISK_ANALYSIS_PATH)

    def _prepare_features(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        detection_df = self.load_detection_log()
        detection_df["confidence"] = pd.to_numeric(detection_df["confidence"], errors="coerce")
        detection_df["count"] = pd.to_numeric(detection_df["count"], errors="coerce")
        detection_df = detection_df.dropna(subset=["count"])
        forecast_df = self.load_forecast()
        risk_df = self.load_risk_metrics()

        if detection_df.empty:
            raise ValueError("Detection log is empty. Collect detections to build recommendations.")

        agg = (
            detection_df.groupby("product_name")
            .agg(count_sum=("count", "sum"), confidence_mean=("confidence", "mean"))
            .reset_index()
        )
        forecast_mean = float(forecast_df["predicted_units"].mean())
        shortage_prob = float(risk_df["shortage_prob"].mean())
        overstock_prob = float(risk_df["overstock_prob"].mean())

        agg["forecast_mean"] = forecast_mean
        agg["shortage_prob"] = shortage_prob
        agg["overstock_prob"] = overstock_prob

        reorder_target = agg["forecast_mean"] * (1 + shortage_prob) - agg["count_sum"]
        reorder_target = reorder_target.clip(lower=0)
        safety_target = agg["count_sum"] * 0.3 + shortage_prob * 10

        features = agg[["product_name", "count_sum", "confidence_mean", "forecast_mean", "shortage_prob", "overstock_prob"]]
        return features, reorder_target, safety_target

    def train(self) -> OptimizationResult:
        features, reorder_target, safety_target = self._prepare_features()
        feature_values = features.drop(columns=["product_name"])
        self.reorder_model.fit(feature_values, reorder_target)
        self.safety_model.fit(feature_values, safety_target)

        joblib.dump(self.reorder_model, ML_MODEL_PATH)
        joblib.dump(self.safety_model, SAFETY_MODEL_PATH)

        recommendations = self._generate_recommendations(features)
        recommendations.to_csv(RECOMMENDATIONS_PATH, index=False)
        return OptimizationResult(recommendations, (ML_MODEL_PATH, SAFETY_MODEL_PATH))

    def _generate_recommendations(self, features: pd.DataFrame) -> pd.DataFrame:
        feature_values = features.drop(columns=["product_name"])
        reorder_predictions = self.reorder_model.predict(feature_values)
        safety_predictions = self.safety_model.predict(feature_values)

        risk_df = self.load_risk_metrics()
        shortage_prob = risk_df["shortage_prob"].mean()
        overstock_prob = risk_df["overstock_prob"].mean()

        df = pd.DataFrame(
            {
                "product_name": features["product_name"],
                "reorder_qty": np.round(reorder_predictions, 2),
                "safety_stock": np.round(safety_predictions, 2),
                "shortage_prob": shortage_prob,
                "overstock_prob": overstock_prob,
            }
        )
        return df

    def predict(self) -> pd.DataFrame:
        if not ML_MODEL_PATH.exists() or not SAFETY_MODEL_PATH.exists():
            raise FileNotFoundError("Optimizer model not found. Train the optimizer first.")
        reorder_model = joblib.load(ML_MODEL_PATH)
        safety_model = joblib.load(SAFETY_MODEL_PATH)
        self.reorder_model = reorder_model
        self.safety_model = safety_model
        features, _, _ = self._prepare_features()
        recommendations = self._generate_recommendations(features)
        recommendations.to_csv(RECOMMENDATIONS_PATH, index=False)
        return recommendations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory optimization")
    parser.add_argument("command", choices=["train", "predict"], help="Train or predict recommendations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    optimizer = InventoryOptimizer()
    if args.command == "train":
        result = optimizer.train()
        print("Recommendations saved to", RECOMMENDATIONS_PATH)
        print(result.recommendations)
    else:
        recommendations = optimizer.predict()
        print("Recommendations updated")
        print(recommendations)


if __name__ == "__main__":
    main()
