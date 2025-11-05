"""Streamlit dashboard for the AI Retail Intelligence Platform."""
from __future__ import annotations

import glob
from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st

from src.deep_forecast import RetailDemandForecaster, FORECAST_RESULTS_PATH
from src.ml_optimizer import InventoryOptimizer, RECOMMENDATIONS_PATH
from src.risk_simulation import RISK_ANALYSIS_PATH, RiskSimulator

DETECTION_LOG_PATH = Path("data/detection_log.csv")
DETECTION_IMAGE_DIR = Path("img/detections")
RISK_FIGURE_PATH = Path("img/risk_distribution.png")

st.set_page_config(page_title="AI Retail Intelligence", layout="wide")
st.title("🛒 AI Retail Intelligence Platform")


@st.cache_data
def load_detection_log() -> pd.DataFrame:
    if not DETECTION_LOG_PATH.exists():
        return pd.DataFrame(columns=["timestamp", "product_name", "confidence", "count"])
    df = pd.read_csv(DETECTION_LOG_PATH)
    if not df.empty:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
        df["count"] = pd.to_numeric(df["count"], errors="coerce")
    return df


@st.cache_data
def load_forecast_results() -> pd.DataFrame:
    if not FORECAST_RESULTS_PATH.exists():
        return pd.DataFrame(columns=["date", "predicted_units"])
    df = pd.read_csv(FORECAST_RESULTS_PATH, parse_dates=["date"])
    return df


@st.cache_data
def load_risk_analysis() -> pd.DataFrame:
    if not RISK_ANALYSIS_PATH.exists():
        return pd.DataFrame(columns=["scenario", "shortage_prob", "overstock_prob", "expected_loss"])
    return pd.read_csv(RISK_ANALYSIS_PATH)


@st.cache_data
def load_recommendations() -> pd.DataFrame:
    if not RECOMMENDATIONS_PATH.exists():
        return pd.DataFrame(
            columns=["product_name", "reorder_qty", "safety_stock", "shortage_prob", "overstock_prob"]
        )
    return pd.read_csv(RECOMMENDATIONS_PATH)


def get_latest_detection_image() -> str | None:
    images: List[str] = glob.glob(str(DETECTION_IMAGE_DIR / "*.jpg"))
    if not images:
        return None
    return max(images, key=Path)


def retrain_pipeline() -> None:
    with st.spinner("Retraining models..."):
        forecaster = RetailDemandForecaster()
        forecast = forecaster.train()
        simulator = RiskSimulator(inventory_level=sum(forecast.values))
        simulator.simulate()
        optimizer = InventoryOptimizer()
        optimizer.train()
    st.success("Pipeline retrained successfully!")


tab1, tab2, tab3, tab4 = st.tabs([
    "🧃 Camera View",
    "📈 Trend Forecast",
    "📊 Risk Simulation",
    "🧩 Optimization",
])

with tab1:
    st.header("Realtime Product Identification")
    st.write(
        "Run `python main.py --source <camera_url>` in a separate terminal to stream detections."
    )
    detection_log = load_detection_log()
    detection_log = detection_log.dropna(subset=["count"])
    if detection_log.empty:
        st.info("No detections logged yet. Start the realtime identifier to populate data.")
    else:
        summary = detection_log.groupby("product_name").agg(
            total_count=("count", "sum"), avg_confidence=("confidence", "mean")
        )
        st.subheader("Detection Summary")
        st.dataframe(summary)

    latest_image = get_latest_detection_image()
    if latest_image:
        st.subheader("Latest Detection Snapshot")
        st.image(latest_image, use_column_width=True)
    else:
        st.caption("Bounding box snapshots will appear here when `--save` is used.")

with tab2:
    st.header("Demand Forecast")
    forecast_df = load_forecast_results()
    if forecast_df.empty:
        st.warning("No forecast results found. Train the model or use the retrain button below.")
    else:
        fig = px.line(forecast_df, x="date", y="predicted_units", title="Forecasted Demand")
        st.plotly_chart(fig, use_container_width=True)
        forecaster = RetailDemandForecaster()
        try:
            trend, interval = forecaster.predict_trend()
            st.metric("Trend", trend.capitalize(), delta=f"Confidence interval: {interval[0]:.1f} - {interval[1]:.1f}")
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

with tab3:
    st.header("Inventory Risk Simulation")
    risk_df = load_risk_analysis()
    if risk_df.empty:
        st.warning("Run the risk simulation to view metrics.")
    else:
        st.dataframe(risk_df)
    if RISK_FIGURE_PATH.exists():
        st.image(str(RISK_FIGURE_PATH), caption="Demand distribution", use_column_width=True)
    else:
        st.caption("Run `python src/risk_simulation.py` to generate the Monte Carlo visualization.")

with tab4:
    st.header("Inventory Optimization Recommendations")
    recommendations = load_recommendations()
    if recommendations.empty:
        st.info("No recommendations available yet. Train the optimizer to generate them.")
    else:
        st.dataframe(recommendations)

st.divider()
if st.button("Retrain Model"):
    retrain_pipeline()
