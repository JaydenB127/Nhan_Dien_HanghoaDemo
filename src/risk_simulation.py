"""Monte Carlo simulation utilities for inventory risk analysis."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.deep_forecast import ForecastResult, predict_trend

RISK_OUTPUT = Path("data/risk_analysis.csv")
RISK_PLOT = Path("img/risk_distribution.png")


def run_risk_simulation(
    forecast: ForecastResult,
    inventory_level: float | None = None,
    simulations: int = 1000,
    demand_std_ratio: float = 0.15,
    output_path: Path = RISK_OUTPUT,
    plot_path: Path = RISK_PLOT,
) -> Dict[str, float]:
    """Execute a Monte Carlo simulation using the demand forecast."""
    if forecast.forecast.empty:
        raise ValueError("Forecast data is empty. Train the forecast model first.")
    expected = forecast.forecast["expected_demand"].to_numpy()
    if inventory_level is None:
        inventory_level = float(expected.sum())

    std_dev = np.maximum(expected * demand_std_ratio, 1.0)
    samples = np.random.normal(loc=expected, scale=std_dev, size=(simulations, len(expected)))
    samples = np.clip(samples, 0, None)
    total_demand = samples.sum(axis=1)

    shortage = total_demand > inventory_level
    overstock = total_demand < inventory_level * 0.9

    shortage_prob = float(np.mean(shortage))
    overstock_prob = float(np.mean(overstock))
    expected_loss = float(np.mean(np.where(shortage, total_demand - inventory_level, inventory_level - total_demand)))

    summary = {
        "date": pd.Timestamp.utcnow().date().isoformat(),
        "expected_demand": float(expected.sum()),
        "simulated_demand": float(total_demand.mean()),
        "shortage_prob": shortage_prob,
        "overstock_prob": overstock_prob,
        "expected_loss": expected_loss,
    }

    save_summary(summary, output_path)
    plot_distribution(total_demand, inventory_level, plot_path)
    return summary


def save_summary(summary: Dict[str, float], path: Path) -> None:
    """Persist risk summary statistics to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([summary])
    if path.exists():
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(path, index=False)


def plot_distribution(simulated: np.ndarray, inventory_level: float, output_path: Path) -> None:
    """Create a histogram plot of simulated demand distributions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(simulated, bins=30, alpha=0.7, color="#2b8a3e")
    ax.axvline(inventory_level, color="red", linestyle="--", label="Inventory level")
    ax.set_xlabel("Total demand")
    ax.set_ylabel("Frequency")
    ax.set_title("Monte Carlo demand distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def cli() -> None:
    """Command-line wrapper for running the simulation."""
    parser = argparse.ArgumentParser(description="Run Monte Carlo risk simulations")
    parser.add_argument(
        "--inventory",
        type=float,
        default=None,
        help="Current inventory level. Defaults to forecasted total demand.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=1000,
        help="Number of Monte Carlo simulations to perform.",
    )
    args = parser.parse_args()

    try:
        forecast = predict_trend()
    except FileNotFoundError as exc:
        raise SystemExit(
            "Forecast model not available. Run 'python src/deep_forecast.py' first."
        ) from exc

    try:
        summary = run_risk_simulation(
            forecast,
            inventory_level=args.inventory,
            simulations=args.simulations,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(summary)


if __name__ == "__main__":
    cli()
