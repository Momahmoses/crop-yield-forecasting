"""
Agricultural Crop Yield Forecasting & Farmland Monitoring
==========================================================
Interactive dashboard for predicting crop yields across North-Central Nigeria
using satellite-derived NDVI, weather patterns, and farm management data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_generator import generate_crop_dataset
from model import (
    FEATURE_COLS, METRICS_PATH, MODEL_PATH, ENCODER_PATH,
    load_model, predict_yield, save_model, train,
)

st.set_page_config(
    page_title="Crop Yield Forecasting | Nigeria",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.yield-high   { color: #27AE60; font-weight: 700; }
.yield-medium { color: #F39C12; font-weight: 700; }
.yield-low    { color: #E74C3C; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

CROPS = ["Maize", "Sorghum", "Millet", "Cassava", "Yam", "Rice", "Groundnut"]


@st.cache_resource(show_spinner="Training yield model…")
def get_model_and_data():
    df = generate_crop_dataset()
    if MODEL_PATH.exists() and METRICS_PATH.exists():
        pipeline, le = load_model()
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    else:
        pipeline, le, metrics = train(df)
        save_model(pipeline, le, metrics)
    return pipeline, le, metrics, df


pipeline, le, metrics, df = get_model_and_data()

with st.sidebar:
    st.title("🌾 Yield Forecasting")
    st.caption("North-Central Nigeria Farm Intelligence")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Overview", "Yield Map", "Forecast a Farm", "Model Performance"],
        label_visibility="collapsed",
    )
    st.divider()
    crop_filter = st.multiselect("Filter by Crop", CROPS, default=CROPS)
    season_filter = st.multiselect("Filter by Season", sorted(df["season"].unique()), default=sorted(df["season"].unique()))

df_f = df[(df["crop"].isin(crop_filter)) & (df["season"].isin(season_filter))]

# ── OVERVIEW ──────────────────────────────────────────────────────────────────
if page == "Overview":
    st.title("Agricultural Crop Yield Forecasting & Farmland Monitoring")
    st.markdown("Satellite imagery and ML-powered yield prediction for smallholder farmers across North-Central Nigeria.")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Farm Plots Analyzed", f"{len(df_f):,}")
    c2.metric("Mean Yield (kg/ha)", f"{df_f['yield_kg_per_ha'].mean():,.0f}")
    c3.metric("Model R² Score", f"{metrics['r2_train']:.4f}")
    c4.metric("LOO MAE (kg/ha)", f"{metrics['loo_mae_mean']:,.0f}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(df_f, x="crop", y="yield_kg_per_ha", color="crop",
                     title="Yield Distribution by Crop", height=420,
                     labels={"yield_kg_per_ha": "Yield (kg/ha)"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        trend = df_f.groupby(["season", "crop"])["yield_kg_per_ha"].mean().reset_index()
        fig2 = px.line(trend, x="season", y="yield_kg_per_ha", color="crop",
                       title="Yield Trends by Season", height=420,
                       labels={"yield_kg_per_ha": "Mean Yield (kg/ha)"})
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.scatter(df_f.sample(min(1500, len(df_f)), random_state=42),
                          x="ndvi_mean", y="yield_kg_per_ha", color="crop",
                          title="NDVI vs Yield", opacity=0.5, height=380,
                          labels={"ndvi_mean": "Mean NDVI", "yield_kg_per_ha": "Yield (kg/ha)"})
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        fi = metrics["feature_importance"]
        fi_sorted = sorted(fi.items(), key=lambda x: x[1])
        fig4 = go.Figure(go.Bar(
            x=[v for _, v in fi_sorted], y=[k for k, _ in fi_sorted],
            orientation="h", marker_color="#3498DB",
        ))
        fig4.update_layout(title="Feature Importance", height=380, margin={"l": 140})
        st.plotly_chart(fig4, use_container_width=True)

# ── YIELD MAP ────────────────────────────────────────────────────────────────
elif page == "Yield Map":
    st.title("Geographic Yield Distribution")
    sample = df_f.sample(min(1000, len(df_f)), random_state=42).copy()
    sample["yield_norm"] = (sample["yield_kg_per_ha"] - sample["yield_kg_per_ha"].min()) / \
                           (sample["yield_kg_per_ha"].max() - sample["yield_kg_per_ha"].min() + 1)
    fig = px.scatter_mapbox(
        sample, lat="latitude", lon="longitude",
        color="yield_kg_per_ha", size="yield_norm", size_max=12,
        color_continuous_scale="YlGn",
        hover_name="state",
        hover_data={"crop": True, "yield_kg_per_ha": ":.0f", "season": True, "latitude": False, "longitude": False, "yield_norm": False},
        mapbox_style="carto-positron", zoom=5,
        center={"lat": 10.5, "lon": 7.5},
        title="Farm-Level Yield Map — North-Central Nigeria", height=540,
    )
    fig.update_layout(coloraxis_colorbar_title="Yield (kg/ha)")
    st.plotly_chart(fig, use_container_width=True)

# ── FORECAST A FARM ──────────────────────────────────────────────────────────
elif page == "Forecast a Farm":
    st.title("Forecast Yield for a Custom Farm")
    st.markdown("Enter farm characteristics to predict expected yield per hectare.")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        crop = st.selectbox("Crop Type", CROPS)
        ndvi_mean = st.slider("Mean NDVI (satellite vegetation index)", 0.1, 0.9, 0.55, 0.01)
        ndvi_max = st.slider("Peak NDVI", float(ndvi_mean), 0.95, min(ndvi_mean + 0.1, 0.95), 0.01)
        rainfall_mm = st.slider("Growing Season Rainfall (mm)", 100.0, 1200.0, 650.0, 10.0)
        temp_mean_c = st.slider("Mean Temperature (°C)", 18.0, 38.0, 28.0, 0.5)
        temp_max_c = st.slider("Max Temperature (°C)", float(temp_mean_c), 45.0, min(temp_mean_c + 6, 45.0), 0.5)
    with col2:
        humidity_pct = st.slider("Relative Humidity (%)", 20.0, 95.0, 65.0, 1.0)
        growing_days = st.slider("Growing Period (days)", 60, 180, 120, 5)
        fertilizer_kg = st.number_input("Fertilizer Applied (kg/ha)", 0.0, 300.0, 60.0, 5.0)
        soil_ph = st.slider("Soil pH", 4.5, 8.5, 6.2, 0.1)
        soil_nitrogen = st.slider("Soil Nitrogen (mg/kg)", 5.0, 100.0, 35.0, 1.0)
        irrigation = st.checkbox("Irrigated Farm?", value=False)
        farm_size_ha = st.slider("Farm Size (ha)", 0.5, 15.0, 3.0, 0.5)

    if st.button("Forecast Yield", type="primary"):
        result = predict_yield(pipeline, le, {
            "crop": crop, "ndvi_mean": ndvi_mean, "ndvi_max": ndvi_max,
            "rainfall_mm": rainfall_mm, "temp_mean_c": temp_mean_c,
            "temp_max_c": temp_max_c, "humidity_pct": humidity_pct,
            "growing_days": growing_days, "fertilizer_kg": fertilizer_kg,
            "soil_ph": soil_ph, "soil_nitrogen": soil_nitrogen,
            "irrigation": int(irrigation), "farm_size_ha": farm_size_ha,
        })
        yld = result["yield_kg_per_ha"]
        total = round(yld * farm_size_ha, 1)
        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Yield", f"{yld:,.0f} kg/ha")
        r2.metric("Total Estimated Harvest", f"{total:,.0f} kg")
        r3.metric("Crop", crop)
        st.info(f"At current local market rates, this harvest from {farm_size_ha} ha of {crop} represents approximately **{total:,.0f} kg** of produce.")

# ── MODEL PERFORMANCE ─────────────────────────────────────────────────────────
elif page == "Model Performance":
    st.title("Model Performance — Random Forest Regressor")
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Train R²", f"{metrics['r2_train']:.4f}")
    m2.metric("Train MAE (kg/ha)", f"{metrics['mae_train']:,.1f}")
    m3.metric("Train RMSE (kg/ha)", f"{metrics['rmse_train']:,.1f}")
    m4.metric("LOO-Season MAE", f"{metrics['loo_mae_mean']:,.1f} ± {metrics['loo_mae_std']:,.1f}")

    fi = metrics["feature_importance"]
    fi_sorted = sorted(fi.items(), key=lambda x: x[1])
    fig = go.Figure(go.Bar(
        x=[v for _, v in fi_sorted], y=[k for k, _ in fi_sorted],
        orientation="h", marker_color="#27AE60",
        text=[f"{v:.3f}" for _, v in fi_sorted], textposition="outside",
    ))
    fig.update_layout(title="Feature Importance Breakdown", height=420, margin={"l": 160, "r": 60})
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Full Metrics JSON"):
        st.json(metrics)
