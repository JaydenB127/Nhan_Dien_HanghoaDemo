"""Streamlit dashboard for the Retail Intelligence Platform."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from src.deep_forecast import predict_trend
from src.ml_optimizer import Recommendation, generate_recommendations
from src.risk_simulation import run_simulation

DATA_DIR = Path("data")
IMG_DIR = Path("img")


def _latest_detection_image() -> Path | None:
    detection_dir = IMG_DIR / "detections"
    if not detection_dir.exists():
        return None
    images = sorted(detection_dir.glob("*.jpg"), key=lambda item: item.stat().st_mtime, reverse=True)
    return images[0] if images else None


def _render_retrain_button() -> None:
    if st.sidebar.button("Retrain Model"):
        with st.spinner("Retraining pipeline. This may take a while..."):
            forecast = predict_trend()
            risk_df = run_simulation()
            recommendations = generate_recommendations()
            st.session_state["forecast_result"] = forecast
            st.session_state["risk_df"] = risk_df
            st.session_state["recommendations"] = recommendations
        st.success("Pipeline retrained successfully!")


def _get_forecast_result():
    if "forecast_result" not in st.session_state:
        st.session_state["forecast_result"] = predict_trend()
    return st.session_state["forecast_result"]


def _get_risk_dataframe() -> pd.DataFrame:
    if "risk_df" not in st.session_state:
        st.session_state["risk_df"] = run_simulation()
    return st.session_state["risk_df"]


def _get_recommendations() -> List[Recommendation]:
    if "recommendations" not in st.session_state:
        st.session_state["recommendations"] = generate_recommendations()
    return st.session_state["recommendations"]


def render_camera_tab() -> None:
    st.subheader("Real-time Camera View")
    st.write(
        "Run `python main.py --source 0` or provide an IP camera stream to start the YOLOv8 pipeline."
    )
    latest_image = _latest_detection_image()
    if latest_image:
        st.image(str(latest_image), caption="Latest detection snapshot")
    else:
        st.info("No detection snapshots found yet. Enable the --save option when running main.py to capture frames.")


def render_trend_tab() -> None:
    st.subheader("Trend Forecast")
    forecast = _get_forecast_result()
    plot_path = IMG_DIR / "forecast_trend.png"
    if plot_path.exists():
        st.image(str(plot_path), caption="Demand trend forecast")
    st.metric(
        label="Predicted next-day demand",
        value=f"{forecast.next_value:.2f}",
        delta=forecast.trend,
    )
    st.write(
        "Confidence interval: ({:.2f}, {:.2f})".format(
            forecast.confidence_interval[0], forecast.confidence_interval[1]
        )
    )


def render_risk_tab() -> None:
    st.subheader("Risk Simulation")
    risk_df = _get_risk_dataframe()
    if risk_df.empty:
        st.warning("Risk analysis data not available. Run the simulation first.")
    else:
        st.dataframe(risk_df)
    risk_plot = IMG_DIR / "risk_histogram.png"
    if risk_plot.exists():
        st.image(str(risk_plot), caption="Monte Carlo demand distribution")


def render_optimization_tab() -> None:
    st.subheader("Inventory Optimization")
    recommendations = _get_recommendations()
    if not recommendations:
        st.warning("No recommendations calculated yet.")
    else:
        data = pd.DataFrame([rec.__dict__ for rec in recommendations])
        st.dataframe(data)
    rec_path = DATA_DIR / "recommendations.csv"
    if rec_path.exists():
        st.download_button(
            label="Download recommendations CSV",
            data=rec_path.read_bytes(),
            file_name="recommendations.csv",
        )


def main() -> None:
    st.set_page_config(page_title="AI Retail Intelligence Platform", layout="wide")
    st.title("AI Retail Intelligence Platform")
    _render_retrain_button()
    tab1, tab2, tab3, tab4 = st.tabs([
        "\U0001F964 Camera View",
        "\U0001F4C8 Trend Forecast",
        "\U0001F4CA Risk Simulation",
        "\U0001F9E9 Optimization",
    ])
    with tab1:
        render_camera_tab()
    with tab2:
        render_trend_tab()
    with tab3:
        render_risk_tab()
    with tab4:
        render_optimization_tab()


if __name__ == "__main__":
    main()
