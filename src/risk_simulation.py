"""Monte Carlo risk simulation for inventory management."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.deep_forecast import TrendForecastResult, predict_trend

RISK_OUTPUT_PATH = Path("data/risk_analysis.csv")
RISK_PLOT_PATH = Path("img/risk_histogram.png")


@dataclass
class RiskMetrics:
    """Summary of stock-out risk for a single product."""

    product_name: str
    shortage_prob: float
    overstock_prob: float
    expected_loss: float


class MonteCarloRiskSimulator:
    """Runs Monte Carlo simulations based on forecasted demand."""

    def __init__(self, iterations: int = 1000) -> None:
        self.iterations = iterations

    @staticmethod
    def _load_detection_snapshot(path: Path = Path("data/detection_log.csv")) -> Dict[str, int]:
        if not path.exists() or path.stat().st_size == 0:
            return {"cola": 50, "orange_juice": 40}
        dataframe = pd.read_csv(path)
        if dataframe.empty:
            return {"cola": 50, "orange_juice": 40}
        grouped = dataframe.groupby("product_name")["count"].mean()
        return grouped.fillna(0).astype(int).to_dict()

    def run(self, forecast: TrendForecastResult) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        stock_levels = self._load_detection_snapshot()
        demand_mean = forecast.next_value
        demand_std = max(np.std(forecast.history.values[-7:]), 1.0)
        rng = np.random.default_rng(seed=42)

        metrics: list[RiskMetrics] = []
        distributions: Dict[str, np.ndarray] = {}
        for product, stock in stock_levels.items():
            samples = rng.normal(loc=demand_mean, scale=demand_std, size=self.iterations)
            shortage = samples > stock
            overstock = samples < stock * 0.5
            shortage_prob = float(np.mean(shortage))
            overstock_prob = float(np.mean(overstock))
            loss = np.where(shortage, (samples - stock) * 2.0, (stock - samples) * 0.5)
            expected_loss = float(np.mean(np.clip(loss, 0, None)))
            metrics.append(
                RiskMetrics(
                    product_name=product,
                    shortage_prob=shortage_prob,
                    overstock_prob=overstock_prob,
                    expected_loss=expected_loss,
                )
            )
            distributions[product] = samples
        dataframe = pd.DataFrame([metric.__dict__ for metric in metrics])
        return dataframe, distributions

    @staticmethod
    def save_results(dataframe: pd.DataFrame) -> None:
        RISK_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(RISK_OUTPUT_PATH, index=False)

    @staticmethod
    def plot_distributions(distributions: Dict[str, np.ndarray]) -> None:
        RISK_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 5))
        for product, samples in distributions.items():
            plt.hist(samples, bins=30, alpha=0.5, label=product)
        plt.title("Monte Carlo Demand Simulation")
        plt.xlabel("Simulated demand")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RISK_PLOT_PATH)
        plt.close()


def run_simulation(iterations: int = 1000) -> pd.DataFrame:
    forecast = predict_trend()
    simulator = MonteCarloRiskSimulator(iterations=iterations)
    dataframe, distributions = simulator.run(forecast)
    simulator.save_results(dataframe)
    simulator.plot_distributions(distributions)
    return dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Monte Carlo risk simulations")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of Monte Carlo iterations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_simulation(iterations=args.iterations)
    print(results)


if __name__ == "__main__":
    main()
