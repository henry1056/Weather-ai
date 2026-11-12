import json
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from weather_pipeline import (
    fetch_historical_weather,
    engineer_features,
    build_weather_model,
    train_and_evaluate,
    predict_upcoming_weather,
)

st.set_page_config(
    page_title="Weather-AI Studio",
    page_icon="⛅",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Pre-defined major cities coordinates
PRESET_CITIES = {
    "New York City, USA": (40.7128, -74.0060),
    "London, UK": (51.5074, -0.1278),
    "Tokyo, Japan": (35.6762, 139.6503),
    "Paris, France": (48.8566, 2.3522),
    "Sydney, Australia": (-33.8688, 151.2093),
    "San Francisco, USA": (37.7749, -122.4194),
    "Cairo, Egypt": (30.0444, 31.2357),
    "Custom Coordinates": (0.0, 0.0),
}

st.sidebar.title("⛅ Weather-AI Studio")
st.sidebar.caption("Open-Meteo ERA5 Reanalysis & XGBoost ML Pipeline")

selected_city = st.sidebar.selectbox("Select Preset Location", list(PRESET_CITIES.keys()))

if selected_city != "Custom Coordinates":
    default_lat, default_lon = PRESET_CITIES[selected_city]
else:
    default_lat, default_lon = 40.7128, -74.0060

lat = st.sidebar.number_input("Latitude", value=float(default_lat), format="%.4f")
lon = st.sidebar.number_input("Longitude", value=float(default_lon), format="%.4f")

st.sidebar.subheader("Training Time Horizon")
start_year = st.sidebar.slider("Training Start Year", 2018, 2023, 2020)
end_year = st.sidebar.slider("Training End Year", start_year + 1, 2025, 2024)

start_date_str = f"{start_year}-01-01"
end_date_str = f"{end_year}-12-31"

st.sidebar.markdown("---")
train_button = st.sidebar.button("🚀 Train Model & Run Forecast", type="primary")

st.title("🌦️ Weather-AI: Meteorological ML Studio")
st.markdown(
    "Explore historical ERA5 atmospheric reanalysis, engineer physical/autoregressive features, "
    "train regularized **XGBoost Regressors**, and predict upcoming daily temperatures."
)

if "model_state" not in st.session_state:
    st.session_state.model_state = None

if train_button:
    with st.spinner("Fetching ERA5 historical reanalysis data from Open-Meteo API..."):
        try:
            raw_df = fetch_historical_weather(lat, lon, start_date_str, end_date_str)
            st.session_state.raw_df = raw_df
        except Exception as e:
            st.error(f"Failed to fetch historical data: {e}")
            st.stop()

    with st.spinner("Engineering meteorological features (lags, rolling averages, seasonality)..."):
        try:
            features_df = engineer_features(raw_df, is_training=True)
            st.session_state.features_df = features_df
        except Exception as e:
            st.error(f"Feature engineering failed: {e}")
            st.stop()

    with st.spinner("Training regularized XGBoost Regressor and evaluating on test split..."):
        try:
            model, feature_cols, metrics = train_and_evaluate(features_df)
            forecast = predict_upcoming_weather(
                model=model,
                feature_cols=feature_cols,
                latitude=lat,
                longitude=lon,
                recent_data=raw_df.tail(60),
            )
            st.session_state.model_state = {
                "model": model,
                "feature_cols": feature_cols,
                "metrics": metrics,
                "forecast": forecast,
                "city": selected_city,
            }
            st.success("Model trained and evaluated successfully!")
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

if st.session_state.model_state:
    state = st.session_state.model_state
    metrics = state["metrics"]
    forecast = state["forecast"]
    raw_df = st.session_state.raw_df
    features_df = st.session_state.features_df

    st.subheader(f"Forecast for {state['city']} ({lat:.2f}°, {lon:.2f}°)")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Predicted Mean Temp",
            value=f"{forecast['predicted_mean_temp_c']:.1f} °C",
            delta=f"{forecast['predicted_mean_temp_c'] - forecast['last_observed_temp_c']:.1f} °C vs Yesterday",
        )
    with col2:
        st.metric(
            label="Last Observed Temp",
            value=f"{forecast['last_observed_temp_c']:.1f} °C",
            help=f"Observed on {forecast['last_observed_date']}",
        )
    with col3:
        st.metric(
            label="Test MAE",
            value=f"{metrics['MAE']:.2f} °C",
            delta=f"{(metrics['Persistence_Baseline_MAE'] - metrics['MAE']):.2f} °C better than Persistence",
        )
    with col4:
        st.metric(
            label="Test RMSE",
            value=f"{metrics['RMSE']:.2f} °C",
        )

    tab1, tab2, tab3 = st.tabs(["📈 Historical & Features", "🧠 Feature Importance", "📋 Data Explorer"])

    with tab1:
        st.subheader("Historical Temperature & 30-Day Climate Baseline")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=raw_df.index,
            y=raw_df["temperature_2m_mean"],
            mode="lines",
            name="Daily Mean Temp (°C)",
            line=dict(color="#38bdf8", width=1),
            opacity=0.6,
        ))
        if "temperature_2m_mean_roll_mean_30d" in features_df.columns:
            fig.add_trace(go.Scatter(
                x=features_df.index,
                y=features_df["temperature_2m_mean_roll_mean_30d"],
                mode="lines",
                name="30-Day Rolling Mean (°C)",
                line=dict(color="#f97316", width=2),
            ))
        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            height=400,
            xaxis_title="Date",
            yaxis_title="Temperature (°C)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Top Meteorological Predictors")
        importance_scores = state["model"].feature_importances_
        importance_df = pd.DataFrame({
            "Feature": state["feature_cols"],
            "Importance": importance_scores,
        }).sort_values("Importance", ascending=False).head(12)

        fig_imp = go.Figure(go.Bar(
            x=importance_df["Importance"],
            y=importance_df["Feature"],
            orientation="h",
            marker=dict(color="#06b6d4"),
        ))
        fig_imp.update_layout(
            template="plotly_dark",
            yaxis=dict(autorange="reversed"),
            xaxis_title="Relative Feature Importance (Gain)",
            margin=dict(l=20, r=20, t=20, b=20),
            height=420,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with tab3:
        st.subheader("Raw Data & Extracted Features")
        st.dataframe(features_df.tail(100), use_container_width=True)
else:
    st.info("👈 Configure your location and training range in the sidebar, then click **'Train Model & Run Forecast'** to begin.")
