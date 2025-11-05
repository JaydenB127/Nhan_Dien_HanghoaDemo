"""Monte Carlo risk simulation for inventory planning."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import json

import pandas as pd

from src.deep_forecast import FORECAST_META_PATH, FORECAST_RESULTS_PATH

DATA_DIR = Path("data")
RISK_ANALYSIS_PATH = DATA_DIR / "risk_analysis.csv"
RISK_FIGURE_PATH = Path("img/risk_distribution.png")


@dataclass
class RiskMetrics:
    shortage_prob: float
    overstock_prob: float
    expected_loss: float


class RiskSimulator:
    """Run Monte Carlo simulations using forecasted demand."""

    def __init__(
        self,
        inventory_level: float = 1000.0,
        shortage_cost: float = 5.0,
        overstock_cost: float = 2.0,
    ) -> None:
        self.inventory_level = inventory_level
        self.shortage_cost = shortage_cost
        self.overstock_cost = overstock_cost
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def load_forecast(self) -> pd.DataFrame:
        if not FORECAST_RESULTS_PATH.exists():
            raise FileNotFoundError("Forecast results not found. Train the forecasting model first.")
        return pd.read_csv(FORECAST_RESULTS_PATH)

    def _load_forecast_std(self) -> float:
        if not FORECAST_META_PATH.exists():
            return 10.0
        with FORECAST_META_PATH.open() as file:
            meta = json.load(file)
        return float(meta.get("residual_std", 10.0))

    def simulate(self, num_simulations: int = 1000) -> RiskMetrics:
        forecast_df = self.load_forecast()
        residual_std = self._load_forecast_std()
        means = forecast_df["predicted_units"].values
        inventory_level = self.inventory_level or float(means.sum())

        total_demands = []
        losses = []

        for _ in range(num_simulations):
            simulated = np.random.normal(loc=means, scale=residual_std)
            simulated = np.clip(simulated, a_min=0, a_max=None)
            total_demand = float(simulated.sum())
            shortage = max(total_demand - inventory_level, 0.0)
            overstock = max(inventory_level - total_demand, 0.0)
            loss = shortage * self.shortage_cost + overstock * self.overstock_cost
            total_demands.append(total_demand)
            losses.append(loss)

        total_demands_arr = np.array(total_demands)
        losses_arr = np.array(losses)

        shortage_prob = float(np.mean(total_demands_arr > inventory_level))
        overstock_prob = float(np.mean(total_demands_arr < inventory_level))
        expected_loss = float(np.mean(losses_arr))

        metrics = RiskMetrics(shortage_prob, overstock_prob, expected_loss)
        self._save_results(metrics)
        self._plot_distribution(total_demands_arr)
        return metrics

    def _save_results(self, metrics: RiskMetrics) -> None:
        df = pd.DataFrame(
            {
                "scenario": ["monte_carlo"],
                "shortage_prob": [metrics.shortage_prob],
                "overstock_prob": [metrics.overstock_prob],
                "expected_loss": [metrics.expected_loss],
            }
        )
        df.to_csv(RISK_ANALYSIS_PATH, index=False)

    def _plot_distribution(self, total_demands: np.ndarray) -> None:
        RISK_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.hist(total_demands, bins=30, alpha=0.7, color="#1f77b4")
        plt.title("Monte Carlo Demand Distribution")
        plt.xlabel("Total Demand")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(RISK_FIGURE_PATH)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory risk simulation")
    parser.add_argument("--inventory", type=float, default=1000.0, help="Available inventory for the horizon")
    parser.add_argument("--shortage-cost", type=float, default=5.0, help="Cost per unit short")
    parser.add_argument("--overstock-cost", type=float, default=2.0, help="Cost per unit overstock")
    parser.add_argument("--runs", type=int, default=1000, help="Number of Monte Carlo runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    simulator = RiskSimulator(
        inventory_level=args.inventory,
        shortage_cost=args.shortage_cost,
        overstock_cost=args.overstock_cost,
    )
    metrics = simulator.simulate(num_simulations=args.runs)
    print(
        "Shortage prob: {:.2%}, Overstock prob: {:.2%}, Expected loss: {:.2f}".format(
            metrics.shortage_prob, metrics.overstock_prob, metrics.expected_loss
        )
    )


if __name__ == "__main__":
    main()
