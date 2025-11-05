"""Machine learning based stock optimization."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.deep_forecast import predict_trend
from src.risk_simulation import run_simulation

OPTIMIZER_MODEL_PATH = Path("models/ml_optimizer.pkl")
RECOMMENDATION_PATH = Path("data/recommendations.csv")


@dataclass
class Recommendation:
    product_name: str
    reorder_qty: float
    safety_stock: float
    shortage_prob: float
    overstock_prob: float


class InventoryOptimizer:
    """Random Forest based optimizer for reorder recommendations."""

    def __init__(self) -> None:
        self.model: RandomForestRegressor | None = None

    @staticmethod
    def _generate_synthetic_training(samples: int = 500) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(21)
        features, targets = [], []
        for _ in range(samples):
            demand = rng.normal(120, 35)
            inventory = rng.uniform(20, 140)
            shortage_prob = rng.uniform(0, 1)
            overstock_prob = rng.uniform(0, 1)
            forecast = demand + rng.normal(0, 10)
            feature_vector = [demand, inventory, forecast, shortage_prob, overstock_prob]
            reorder_qty = max(forecast - inventory, 0)
            safety_stock = max(forecast * 0.2 + shortage_prob * 15, 5)
            features.append(feature_vector)
            targets.append([reorder_qty, safety_stock])
        return np.array(features, dtype=np.float32), np.array(targets, dtype=np.float32)

    def train(self) -> None:
        x_train, y_train = self._generate_synthetic_training()
        self.model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.model.fit(x_train, y_train)
        OPTIMIZER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, OPTIMIZER_MODEL_PATH)

    def load(self) -> None:
        self.model = joblib.load(OPTIMIZER_MODEL_PATH)

    def ensure_model(self) -> None:
        if self.model is None:
            if OPTIMIZER_MODEL_PATH.exists():
                self.load()
            else:
                self.train()

    def recommend(self, features: pd.DataFrame) -> List[Recommendation]:
        self.ensure_model()
        assert self.model is not None, "Model not available"
        predictions = self.model.predict(features.values)
        recommendations: List[Recommendation] = []
        for idx, product in enumerate(features.index):
            reorder_qty, safety_stock = predictions[idx]
            shortage_prob = float(features.loc[product, "shortage_prob"])
            overstock_prob = float(features.loc[product, "overstock_prob"])
            recommendations.append(
                Recommendation(
                    product_name=product,
                    reorder_qty=float(max(reorder_qty, 0)),
                    safety_stock=float(max(safety_stock, 0)),
                    shortage_prob=shortage_prob,
                    overstock_prob=overstock_prob,
                )
            )
        return recommendations


def _prepare_feature_table(risk_dataframe: pd.DataFrame, forecast_value: float, detection_log: Path) -> pd.DataFrame:
    inventory_snapshot = _load_inventory_counts(detection_log)
    rows = []
    for _, row in risk_dataframe.iterrows():
        product = row["product_name"]
        inventory = inventory_snapshot.get(product, float(np.mean(list(inventory_snapshot.values()) or [100.0])))
        rows.append(
            {
                "product_name": product,
                "demand": forecast_value,
                "inventory": inventory,
                "forecast": forecast_value,
                "shortage_prob": row["shortage_prob"],
                "overstock_prob": row["overstock_prob"],
            }
        )
    feature_df = pd.DataFrame(rows).set_index("product_name")
    return feature_df


def _load_inventory_counts(path: Path) -> Dict[str, float]:
    if not path.exists() or path.stat().st_size == 0:
        return {"cola": 60.0, "orange_juice": 45.0}
    dataframe = pd.read_csv(path)
    if dataframe.empty:
        return {"cola": 60.0, "orange_juice": 45.0}
    grouped = (
        dataframe.sort_values("timestamp")
        .groupby("product_name")
        .tail(5)
        .groupby("product_name")["count"]
        .mean()
    )
    return grouped.to_dict()


def generate_recommendations() -> List[Recommendation]:
    forecast = predict_trend()
    risk_df = run_simulation()
    features = _prepare_feature_table(risk_df, forecast.next_value, Path("data/detection_log.csv"))
    optimizer = InventoryOptimizer()
    recommendations = optimizer.recommend(features)
    _persist_recommendations(recommendations)
    return recommendations


def _persist_recommendations(recommendations: List[Recommendation]) -> None:
    RECOMMENDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame([rec.__dict__ for rec in recommendations])
    dataframe.to_csv(RECOMMENDATION_PATH, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize stock levels using machine learning")
    parser.add_argument("--train-only", action="store_true", help="Only train the Random Forest model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    optimizer = InventoryOptimizer()
    if args.train_only:
        optimizer.train()
        print("Optimizer model trained and saved.")
        return
    recommendations = generate_recommendations()
    for recommendation in recommendations:
        print(recommendation)


if __name__ == "__main__":
    main()
