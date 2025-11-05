"""Streamlit dashboard for the AI Retail Intelligence Platform."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import plotly.express as px
import streamlit as st

from src.deep_forecast import ForecastResult, predict_trend, train_forecast_model
from src.ml_optimizer import train_optimizer_model
from src.risk_simulation import run_risk_simulation

DATA_DIR = Path("data")
IMAGE_DIR = Path("img/detections")


st.set_page_config(page_title="AI Retail Intelligence", layout="wide")
st.title("🛒 AI Retail Intelligence Platform")


def load_csv(path: Path) -> pd.DataFrame:
    """Safely load a CSV file into a DataFrame."""
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def render_camera_view() -> None:
    st.subheader("🧃 Camera View")
    st.write("Run `python main.py --source <stream>` to start the detection pipeline.")

    log_df = load_csv(DATA_DIR / "detection_log.csv")
    if log_df.empty:
        st.info("No detections logged yet.")
    else:
        st.dataframe(log_df.tail(20), use_container_width=True)

    image_files = sorted(IMAGE_DIR.glob("*.jpg"))
    if image_files:
        st.image(str(image_files[-1]), caption="Latest detection frame", use_column_width=True)
    else:
        st.caption("Annotated frames will appear here when --save is enabled.")


def render_forecast_tab(forecast: ForecastResult | None) -> ForecastResult | None:
    st.subheader("📈 Trend Forecast")
    if forecast is None:
        try:
            forecast = predict_trend()
        except FileNotFoundError:
            st.warning("Forecast model not found. Train it using the button above.")
            return None

    history_df = forecast.history
    forecast_df = forecast.forecast
    combined = pd.concat(
        [
            history_df.assign(segment="Historical", value=history_df["total_quantity"]),
            forecast_df.assign(segment="Forecast", value=forecast_df["expected_demand"]),
        ]
    )
    fig = px.line(combined, x="date", y="value", color="segment", title="Demand Forecast")
    st.plotly_chart(fig, use_container_width=True)

    st.metric(
        "Predicted trend",
        forecast.trend.capitalize(),
        f"Confidence interval: {forecast.confidence_interval[0]:.1f} - {forecast.confidence_interval[1]:.1f}",
    )
    return forecast


def render_risk_tab(forecast: ForecastResult | None) -> Dict[str, float] | None:
    st.subheader("📊 Risk Simulation")
    if forecast is None:
        st.info("Forecast required to run risk simulation.")
        return None

    risk_df = load_csv(DATA_DIR / "risk_analysis.csv")
    latest_summary: Dict[str, float] | None = None
    if not risk_df.empty:
        latest_summary = risk_df.iloc[-1].to_dict()
        fig = px.bar(risk_df, x="date", y=["shortage_prob", "overstock_prob"], barmode="group")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(risk_df.tail(10), use_container_width=True)
    else:
        st.caption("Run the risk simulation to populate this section.")

    if st.button("Run Monte Carlo Simulation", key="run_simulation"):
        summary = run_risk_simulation(forecast)
        st.success("Simulation completed.")
        latest_summary = summary
        risk_df = load_csv(DATA_DIR / "risk_analysis.csv")
        if not risk_df.empty:
            st.dataframe(risk_df.tail(10), use_container_width=True)

    if latest_summary:
        st.metric("Shortage probability", f"{latest_summary['shortage_prob']*100:.1f}%")
        st.metric("Overstock probability", f"{latest_summary['overstock_prob']*100:.1f}%")
    return latest_summary


def render_optimizer_tab(
    forecast: ForecastResult | None,
    risk_summary: Dict[str, float] | None,
) -> None:
    st.subheader("🧩 Optimization")
    recommendations = load_csv(DATA_DIR / "recommendations.csv")

    if st.button("Generate Recommendations", key="run_optimizer"):
        if forecast is None or risk_summary is None:
            st.warning("Forecast and risk summary required before optimisation.")
        else:
            try:
                recommendations = train_optimizer_model(forecast=forecast, risk_summary=risk_summary)
                st.success("Recommendations updated.")
            except FileNotFoundError as exc:
                st.error(str(exc))
                return
            except ValueError as exc:
                st.error(str(exc))
                return

    if recommendations.empty:
        st.info("No recommendations available yet.")
    else:
        st.dataframe(recommendations, use_container_width=True)


with st.sidebar:
    st.header("Controls")
    if st.button("Retrain Model", key="retrain_models"):
        try:
            forecast_result = train_forecast_model()
            risk_summary_sidebar = run_risk_simulation(forecast_result)
            train_optimizer_model(forecast=forecast_result, risk_summary=risk_summary_sidebar)
            st.success("Models retrained successfully.")
        except Exception as exc:  # pragma: no cover - UI feedback
            st.error(f"Model retraining failed: {exc}")


forecast_state: ForecastResult | None = None
risk_state: Dict[str, float] | None = None

tabs = st.tabs(["🧃 Camera View", "📈 Trend Forecast", "📊 Risk Simulation", "🧩 Optimization"])
camera_tab, forecast_tab, risk_tab, optimizer_tab = tabs

with camera_tab:
    render_camera_view()

with forecast_tab:
    forecast_state = render_forecast_tab(forecast_state)

with risk_tab:
    if forecast_state is not None:
        risk_state = render_risk_tab(forecast_state)
    else:
        render_risk_tab(None)

with optimizer_tab:
    render_optimizer_tab(forecast_state, risk_state)


if __name__ == "__main__":
    import os
    import subprocess
    import sys

    if os.environ.get("RUNNING_STREAMLIT_SERVER") != "1":
        os.environ["RUNNING_STREAMLIT_SERVER"] = "1"
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())],
            check=True,
        )
